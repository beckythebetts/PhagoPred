"""
Linear baseline: a regression of the output on the last ``window`` frames of
every feature (a distributed-lag model),

    output(L) = b + sum_{f, l < window} w[f, l] x_f(L - 1 - l)

i.e. a causal Conv1d with a single ``window``-frame kernel and no
nonlinearity, read at the last valid frame. With the MSE loss it is linear
regression, with BCE logistic regression. The correctly specified model for
targets that are linear filters of the features (the linear graph
scenarios); the reference the CNN / LSTM are compared with.
"""
import torch
import torch.nn as nn

from PhagoPred.utils.logger import get_logger
from .base import SurvivalModel

log = get_logger()


class LinearSurvival(SurvivalModel):

    def __init__(self, num_features: int, num_bins: int, window: int,
                 **kwargs):
        """
        Args:
            num_features: number of input features per timestep
            num_bins: number of outputs
            window: frames before the landmark the output reads. Frames
                before a sequence's start are zero (the normalised mean).
        """
        super().__init__(num_bins)
        self.num_features = num_features
        self.window = window
        self.conv = nn.Conv1d(num_features, num_bins, window)
        # Convex problem: start from the zero model.
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        num_params = sum(p.numel() for p in self.parameters())
        log.info(f'Linear model initialised with {num_params:,} parameters '
                 f'({window}-frame window)')

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, features) -> output at every frame
        (batch, seq_len, num_bins), each reading only frames <= its own."""
        x = torch.nn.functional.pad(x.permute(0, 2, 1), (self.window - 1, 0))
        return self.conv(x).permute(0, 2, 1)

    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor,
                return_attention: bool = False,
                mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, num_features)
            lengths: (batch,) - valid sequence lengths
            return_attention: also return (batch, seq_len) weights, 1 at the
                frame read (for API compatibility with the attention models)
            mask: unused, for API compatibility

        Returns:
            outputs: (batch, num_bins), at each sequence's last valid frame
        """
        out = self._encode(x)
        last = (lengths.to(out.device) - 1).clamp(min=0).long()
        outputs = out[torch.arange(len(x), device=out.device), last]
        if return_attention:
            weights = torch.zeros(x.shape[:2], device=out.device)
            weights[torch.arange(len(x), device=out.device), last] = 1.0
            return outputs, weights
        return outputs

    def forward_dense(self,
                      x: torch.Tensor,
                      lengths: torch.Tensor,
                      mask: torch.Tensor = None) -> torch.Tensor:
        """Per-timestep outputs (batch, seq_len, num_bins); see SurvivalModel."""
        return self._encode(x)
