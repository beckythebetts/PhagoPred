"""
Temporal CNN-based survival model.
Supports modular attention mechanisms.
"""
import torch
import torch.nn as nn

from PhagoPred.survival_v2.configs.attention import AttentionCfg
from PhagoPred.utils.logger import get_logger
from .base import SurvivalModel, build_fc_layers
from ..attention.mechanisms import get_attention_mechanism, LastPooling

log = get_logger()


class CNNSurvival(SurvivalModel):
    """
    Temporal CNN + Attention + FC for survival prediction.
    Uses dilated convolutions to capture long-range temporal dependencies.
    """

    def __init__(
            self,
            num_features: int,
            num_bins: int,
            num_channels: list[int],
            kernel_sizes: list[int],
            dilations: list[int],
            fc_layers: list[int],
            #  attention_type: str,
            attention_config: AttentionCfg = None,
            causal: bool = False,
            **kwargs):
        """
        Args:
            num_features: number of input features per timestep
            num_bins: number of discrete time bins
            num_channels: output channels for each conv layer
            kernel_sizes: kernel size for each conv layer
            dilations: dilation for each conv layer
            fc_layers: hidden layers for survival prediction head
            attention_type: 'vector', 'fc', 'multihead', 'mean_pool', 'last_pool'
            attention_config: dict of kwargs for attention mechanism
            causal: if True, left-pad each conv so output at t only depends
                on inputs <= t (independent of end-padding in the batch)
        """
        super().__init__(num_bins)

        self.num_features = num_features
        self.causal = causal

        # Validate conv layer specs
        if not (len(num_channels) == len(kernel_sizes) == len(dilations)):
            raise ValueError(
                "num_channels, kernel_sizes, and dilations must have same length"
            )

        # Convolutional layers
        self.cn_layers = nn.ModuleList()
        self.left_pads = []
        in_channels = num_features
        for out_channels, kernel_size, dilation in zip(num_channels,
                                                       kernel_sizes,
                                                       dilations):
            self.cn_layers.append(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size,
                          dilation=dilation,
                          padding=0 if causal else 'same'))
            self.left_pads.append((kernel_size - 1) * dilation if causal else 0)
            in_channels = out_channels

        # Attention mechanism
        # if attention_config is None:
        #     attention_config = {}
        # attention_config['embed_dim'] = in_channels
        setattr(attention_config, 'embed_dim', in_channels)
        self.attention = get_attention_mechanism(attention_config)
        #  **attention_config)

        # Survival prediction head
        self.fc = build_fc_layers(input_size=in_channels,
                                  output_size=num_bins,
                                  layer_sizes=fc_layers)

        num_params = sum(p.numel() for p in self.parameters()
                         if p.requires_grad)
        print(
            f"CNN initialised with {num_params:,} parameters\n\tAttention type: {attention_config.attention_type}"
        )
        log.info(
            f"CNN initialised with {num_params:,} parameters\n\tAttention type: {attention_config.attention_type}"
        )

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor,
                return_attention: bool = False,
                mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, num_features)
            lengths: (batch,) - valid sequence lengths
            return_attention: whether to return attention weights
            mask: optional (not used by CNN, included for API compatibility)

        Returns:
            outputs: (batch, num_bins) - survival logits
            attn_weights: (batch, seq_len) - attention weights (if return_attention)
        """
        # log.info(f'Calling forward with \n{x}')
        batch_size, seq_len, num_features = x.shape

        # Padding mask for attention
        padding_mask = torch.arange(seq_len,
                                    device=lengths.device).unsqueeze(0)
        padding_mask = padding_mask < lengths.unsqueeze(1)

        x = self._encode(x)

        # Attention pooling
        context, attn_weights = self.attention(x, padding_mask)

        # Survival prediction
        outputs = self.fc(context)

        if return_attention:
            return outputs, attn_weights
        else:
            return outputs

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Conv stack: (batch, seq_len, features) -> (batch, seq_len, channels)"""
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.permute(0, 2, 1)

        # Apply convolutional layers
        for layer, left_pad in zip(self.cn_layers, self.left_pads):
            if left_pad:
                x = torch.nn.functional.pad(x, (left_pad, 0))
            x = torch.nn.functional.relu(layer(x))

        # Transpose back: (batch, seq_len, channels)
        return x.permute(0, 2, 1)

    def forward_dense(self,
                      x: torch.Tensor,
                      lengths: torch.Tensor,
                      mask: torch.Tensor = None) -> torch.Tensor:
        """Per-timestep logits (batch, seq_len, num_bins); see SurvivalModel."""
        if not self.causal or not isinstance(self.attention, LastPooling):
            raise ValueError(
                'Dense supervision needs a causal CNN with last_pool attention')
        return self.fc(self._encode(x))
