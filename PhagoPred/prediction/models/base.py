from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from PhagoPred.utils import get_logger
from PhagoPred.survival_v2.probability_calib import (
    TemperatureScalingResult,
    VectorScalingResult,
    PlattScalingResult,
    IsotonicScalingResult,
)

log = get_logger()


def build_fc_layers(input_size: int,
                    output_size: int,
                    layer_sizes: list[int],
                    dropout: float = 0.0,
                    activation=nn.ReLU,
                    batch_norm: bool = False,
                    final_layer_activation: bool = False) -> nn.Sequential:
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
        self.calibration: TemperatureScalingResult | VectorScalingResult | PlattScalingResult | IsotonicScalingResult | None = None

    def set_calibration(
        self,
        result: TemperatureScalingResult | VectorScalingResult
        | PlattScalingResult | IsotonicScalingResult,
    ) -> None:
        self.calibration = result

    def get_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        """Return raw logits (pre-sigmoid hazard logits or binary logit)."""
        return outputs

    def _calibrate_logits(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply calibration to hazard logits (survival) or binary logit."""
        if self.calibration is None:
            return outputs
        if isinstance(self.calibration, TemperatureScalingResult):
            return outputs / self.calibration.T
        if isinstance(self.calibration, VectorScalingResult):
            w = torch.tensor(self.calibration.w,
                             dtype=outputs.dtype,
                             device=outputs.device)
            b = torch.tensor(self.calibration.b,
                             dtype=outputs.dtype,
                             device=outputs.device)
            return outputs * w + b
        if isinstance(self.calibration, PlattScalingResult):
            return outputs * self.calibration.a + self.calibration.b
        return outputs

    @abstractmethod
    def forward(self,
                x: torch.Tensor,
                lengths: torch.Tensor,
                return_attention: bool = False,
                mask: torch.Tensor = None):
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
        """Compute hazard for each time bin, then cacluate pmf from hazards"""
        outputs = self._calibrate_logits(outputs)
        h = torch.sigmoid(outputs)  # hazard rates (batch, num_bins)
        log_s = torch.log1p(-h).cumsum(dim=1)  # log S(t) = Σ log(1-h)
        # S(t-1): prepend log S(-1)=0, drop last
        log_s_prev = torch.cat(
            [torch.zeros(h.shape[0], 1, device=h.device), log_s[:, :-1]],
            dim=1)
        return h * torch.exp(log_s_prev)  # PMF(t) = h(t) · S(t-1)

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

    def predict_restricted_survival_time(
            self, outputs: torch.Tensor,
            time_bins: np.ndarray) -> torch.Tensor:
        """Predict the restricted mean survival time (RMST) up to the horizon.

        RMST = area under the survival step-function from 0 to tau, where tau
        is the last element of time_bins.  Each interval [t_i, t_{i+1}]
        contributes S(t_i) * (t_{i+1} - t_i).
        """
        survival = self.predict_survival(outputs)  # (batch, num_bins)

        t = torch.tensor(time_bins,
                         dtype=survival.dtype,
                         device=survival.device)
        # widths of each interval: last bin ends at tau = time_bins[-1]
        widths = torch.diff(t,
                            append=t[-1:] +
                            (t[-1] - t[-2] if len(t) > 1 else t[-1]))
        # widths shape: (num_bins,) — broadcast over batch
        return (survival * widths).sum(dim=1)

    def predict_binary(self, outputs: torch.Tensor) -> torch.Tensor:
        """Predict binary outcome (event within time horizon) from raw outputs.

        Parametric calibration (temperature/vector/Platt) transforms the logit
        and then a sigmoid is applied. Isotonic calibration instead maps the
        sigmoid probability directly through a fitted monotone function, so it
        bypasses the affine-logit path.
        """
        assert self.num_bins == 1, "Binary prediction only valid for num_bins=1"
        if isinstance(self.calibration, IsotonicScalingResult):
            return self._apply_isotonic(torch.sigmoid(outputs))
        outputs = self._calibrate_logits(outputs)
        return torch.nn.functional.sigmoid(outputs)

    def _apply_isotonic(self, probs: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear monotone map matching the fitted IsotonicScalingResult.

        Reproduces sklearn's ``predict`` with ``out_of_bounds='clip'`` (linear
        interpolation between the fitted knots, clamped to the endpoints) while
        keeping everything on ``probs``'s device/dtype.
        """
        cal = self.calibration
        xs = torch.as_tensor(cal.x_thresholds,
                             dtype=probs.dtype,
                             device=probs.device)
        ys = torch.as_tensor(cal.y_thresholds,
                             dtype=probs.dtype,
                             device=probs.device)
        if xs.numel() == 1:  # degenerate fit collapsed to a constant
            return torch.full_like(probs, float(ys[0]))
        p = probs.clamp(xs[0], xs[-1])
        idx = torch.searchsorted(xs, p, right=True).clamp(1, xs.numel() - 1)
        x0, x1 = xs[idx - 1], xs[idx]
        y0, y1 = ys[idx - 1], ys[idx]
        slope = (y1 - y0) / (x1 - x0).clamp_min(1e-12)
        return y0 + slope * (p - x0)
