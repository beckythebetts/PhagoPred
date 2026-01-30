"""
Base class for survival analysis models.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


def build_fc_layers(
    input_size: int,
    output_size: int,
    layer_sizes: list[int] = [],
    dropout: float = 0.0,
    activation = nn.ReLU,
    batch_norm: bool = False,
    final_layer_activation: bool = False
) -> nn.Sequential:
    """
    Build fully-connected layers with optional batch norm and dropout.
    """
    modules = []
    dim_size = input_size

    for layer_size in layer_sizes:
        modules.append(nn.Linear(dim_size, layer_size))
        if batch_norm:
            modules.append(nn.BatchNorm1d(layer_size))
        modules.append(activation())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        dim_size = layer_size

    modules.append(nn.Linear(dim_size, output_size))
    if final_layer_activation:
        modules.append(activation())

    return nn.Sequential(*modules)


class SurvivalModel(nn.Module, ABC):
    """
    Abstract base class for survival analysis models.
    All models output PMF over discrete time bins.
    """

    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        return_attention: bool = False,
        mask: torch.Tensor = None
    ):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, num_features)
            lengths: (batch,) - valid sequence lengths
            return_attention: whether to return attention weights
            mask: optional validity mask for features

        Returns:
            outputs: raw logits (batch, num_bins)
            or (outputs, auxiliary) if model has auxiliary outputs
        """
        pass

    def predict_pmf(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert raw outputs to probability mass function."""
        return torch.nn.functional.softmax(outputs, dim=1)

    def predict_cif(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert raw outputs to cumulative incidence function."""
        pmf = self.predict_pmf(outputs)
        return torch.cumsum(pmf, dim=1)

    def predict_survival(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert raw outputs to survival function."""
        cif = self.predict_cif(outputs)
        return 1.0 - cif

    def predict_median_time(self, outputs: torch.Tensor) -> torch.Tensor:
        """Predict median survival time (time bin where CIF >= 0.5)."""
        cif = self.predict_cif(outputs)
        return torch.argmax((cif >= 0.5).float(), dim=1).float()

    def predict_expected_time(self, outputs: torch.Tensor) -> torch.Tensor:
        """Predict expected survival time (weighted sum over bins)."""
        pmf = self.predict_pmf(outputs)
        time_bins = torch.arange(self.num_bins, device=pmf.device).float()
        return (pmf * time_bins).sum(dim=1)
