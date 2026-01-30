import torch

from PhagoPred.survival_analysis.models import losses

def build_fc_layers(input_size,
                    output_size,
                    layer_sizes: list[int] = [],
                    dropout: float=0.0,
                    activation=torch.nn.ReLU,
                    batch_norm: bool=False,
                    final_layer_activation: bool=False):
    modules = []
    dim_size = input_size
    for layer_size in layer_sizes:
        modules.append(torch.nn.Linear(dim_size, layer_size))
        if batch_norm:
            modules.append(torch.nn.BatchNorm1d(layer_size))
        modules.append(activation())
        if dropout > 0:
            modules.append(torch.nn.Dropout(dropout))
        dim_size = layer_size
    modules.append(torch.nn.Linear(dim_size, output_size))
    if final_layer_activation is True:
        modules.append(activation())
    return torch.nn.Sequential(*modules)


class TemporalCNN(torch.nn.Module):
    """Network consisting of 1D convolutions, outputing predicted time to event distribution"""

    def __init__(self,
                 num_input_features: int,
                 output_size: int,
                 num_channels: list[int] = [64, 64, 64],
                 kernel_sizes: list[int] = [3, 3],
                 dilations: list[int] = [1, 2, 4],
                 attention_layers: list[int] = [64, 64],
                 fc_layers: list[int] = [64, 64],
                 **_
                 ):

        super().__init__()

        if not len(num_channels) == len(kernel_sizes) == len(dilations):
            raise ValueError  ('num_channels, kernel_sizes and dilaitons must be same length')

        self.cn_layers = torch.nn.ModuleList()
        in_channels = num_input_features
        for out_channels, kernel_size, dilation in zip(num_channels, kernel_sizes, dilations):
            self.cn_layers.append(torch.nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding='same'))
            in_channels = out_channels

        self.attention = build_fc_layers(input_size=in_channels, output_size=1, layer_sizes=attention_layers)
        self.attention_vector = torch.nn.Parameter(torch.randn(in_channels))
        self.fc = build_fc_layers(input_size=in_channels, output_size=output_size, layer_sizes=fc_layers)

    def forward(self, x, lengths, return_attention: bool=False) -> torch.Tensor:
        """
        Args
        ----
            x [batchsize, seq_len, features]
        """
        batch_size, seq_len, _ = x.size()
        # Get padding mask
        mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) 
        mask = mask >= (seq_len - lengths).unsqueeze(1) 
        
        x = x.permute(0, 2, 1) # (batchsize, features, seq_length)
        for layer in self.cn_layers:
            x = torch.nn.functional.relu(layer(x))
        x = x.permute(0, 2, 1) # (batchsize, seq_len, features)

        # attn_weights = self.attention(x).squeeze(-1) # (B, T)
        # attn_weights = attn_weights.masked_fill(~mask, -1e9)
        # attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        
        
        attn_weights = torch.einsum("btc,c->bt", x, self.attention_vector)
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)

        context_vector = torch.sum(attn_weights.unsqueeze(-1)*x, dim=1) # (B, channels)
        output = self.fc(context_vector)
        # output = torch.nn.functional.sigmoid(output)

        if return_attention:
            return output, attn_weights
        else:
            return output

# === Model outputs pmf ===
def estimated_pmf(outputs: torch.Tensor) -> torch.Tensor:
    """Probability mass function from Dynamic DeepHit outputs.
    Args:
        outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin

    Returns:
        PMF: (torch.Tensor) - Probabality of event occuring at time indexed by dim=1
    """
    pmf = torch.nn.functional.softmax(outputs, dim=1)
    return pmf

import torch
import torch.nn.functional as F

class TemporalGradCAM:
    """
    Grad-CAM for 1D temporal CNNs.

    Usage:
        cam = TemporalGradCAM(model, target_layer=model.cn_layers[-1])
        outputs = model(x, lengths)
        # Choose target scalar (e.g., probability at time bin 5)
        target = outputs[:, 5].sum()
        cam_map = cam(x, lengths, target)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Forward hook: save activations
        target_layer.register_forward_hook(self.save_activation)
        # Backward hook: save gradients
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, inp, out):
        # out shape: (B, C, T)
        self.activations = out.detach()

    def save_gradient(self, module, grad_in, grad_out):
        # grad_out[0] shape: (B, C, T)
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, lengths: torch.Tensor, target_scalar: torch.Tensor):
        """
        Compute Grad-CAM map over time.

        Args:
            x: (B, T, C) input sequence
            lengths: (B,) sequence lengths
            target_scalar: scalar output to backprop
        Returns:
            cam_map: (B, T) importance over time
        """
        self.model.zero_grad()
        output = self.model(x, lengths)
        target_scalar.backward(retain_graph=True)

        # Get activations and gradients: (B, C, T)
        A = self.activations
        dA = self.gradients

        # Compute channel weights (global average pooling over time)
        weights = dA.mean(dim=2)  # (B, C)

        # Weighted combination of activations
        cam = torch.relu(torch.einsum("bc, bct->bt", weights, A))  # (B, T)

        # Optional: normalize per sequence
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-9)

        return cam

def compute_loss(
    outputs: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
    pmf: torch.Tensor = None,
    mask: torch.Tensor = None,
) -> torch.Tensor:

    t = t.long()

    if pmf is None:
        pmf = estimated_pmf(outputs)
    # assert (pmf >= 0.0).all(), f"PMF has negative values: {pmf}"
    pmf = torch.clamp(pmf, min=0.0, max=1.0)

    negative_log_likelihood, censored_loss, uncesnored_loss = losses.negative_log_likelihood(pmf, t, e)

    ranking_loss = losses.ranking_loss(pmf, t, e)

    loss = negative_log_likelihood + ranking_loss

    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or inf loss encountered:")
        print(f"  Negative Log Likelihood: {negative_log_likelihood.item()}")
        print(f"  Ranking Loss: {ranking_loss.item()}")
        print(f"  NLL censored: {censored_loss.item()}")
        print(f"  NLL uncensored: {uncesnored_loss.item()}")
        print(f"  Inputs: {x}")
        print(f"  Outputs: {outputs}")
        print(f"  PMF: {pmf}")
        print(f"  t: {t}")
        print(f"  e: {e}")
        raise ValueError("NaN or inf loss encountered in compute_loss")

    return loss, negative_log_likelihood, ranking_loss, censored_loss, uncesnored_loss
