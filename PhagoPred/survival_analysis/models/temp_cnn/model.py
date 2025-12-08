import torch
from pytorch_tcn import TCN

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
                 fc_layers: list[int] = [64, 64]
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
        self.fc = build_fc_layers(input_size=in_channels, output_size=output_size, layer_sizes=fc_layers)

    def forward(self, x, return_attention: bool=False) -> torch.Tensor:
        """
        Args
        ----
            x [batchsize, seq_len, features]
        """

        x = x.permute(0, 2, 1) # (batchsize, features, seq_length)
        for layer in self.cn_layers:
            x = layer(x)
        x = x.permute(0, 2, 1) # (batchsize, seq_len, features)

        attn_weights = self.attention(x).squeeze(-1) # (B, T)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1)*x, dim=1) # (B, channels)
        output = self.fc(context_vector)
        output = torch.nn.functional.softmax(output)

        if return_attention:
            return output, attn_weights
        else:
            return output

def estimated_cif(
    outputs: torch.Tensor,
    # t: torch.Tensor
    ) -> torch.Tensor:
    """Cumulative incidence function from Dynamic DeepHit outputs.
    Args:
        outputs: (batch_size, num_time_bins) - predicted probabilities for each time bin
        t: (batch_size,) - true event/censoring times (as indices of time bins)

    Returns:
        CIF: (torch.Tensor) - Probabality of event occuring on or before time indexed by dim=1
    """

    cif = torch.cumsum(outputs, dim=1)
    return cif


def compute_loss(
    outputs: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    x: torch.Tensor = None,
    y: torch.Tensor = None,
) -> torch.Tensor:


    t = t.long()

    cif = estimated_cif(outputs)

    negative_log_likelihood, censored_loss, uncesnored_loss = losses.negative_log_likelihood(outputs, cif, t, e)
    # negative_log_likelihood, censored_loss, uncesnored_loss = losses.soft_NLL(outputs, cif, t, e)

    ranking_loss = losses.ranking_loss(cif, t, e)

    if x is None or y is None:
        prediction_loss = torch.tensor(0.0, device=outputs.device)
    else:
        mask = None

        if mask is None:
            features = x
            mask = torch.ones_like(features, device=features.device)
        else:
            features = x[:, :, :x.size(-1)//2]
            mask = x[:, :, x.size(-1)//2:]

        prediction_loss = losses.prediction_loss(y, features, mask)

    # loss = negative_log_likelihood + ranking_loss + prediction_loss
    loss = negative_log_likelihood + prediction_loss + ranking_loss
    if torch.isnan(loss) or torch.isinf(loss):
        print("NaN or inf loss encountered in final loss!")
        print("NaN or inf loss encountered:")
        print(f"  Negative Log Likelihood: {negative_log_likelihood.item()}")
        print(f"  Ranking Loss: {ranking_loss.item()}")
        print(f"  NLL censored: {censored_loss.item()}")
        print(f"  NLL uncensored: {uncesnored_loss.item()}")
        print(f"  Prediction Loss: {prediction_loss.item()}")
        print(f"  Outputs: {outputs}")
        print(f"  CIF: {cif}")
        print(f"  t: {t}")
        print(f"  e: {e}")
        raise ValueError("NaN or inf loss encountered in compute_loss")

    return loss, negative_log_likelihood, ranking_loss, prediction_loss, censored_loss, uncesnored_loss
