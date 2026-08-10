from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from PhagoPred.survival_v2.probability_calib import (
    CalibrationResult,
    TemperatureScalingResult,
    VectorScalingResult,
    PlattScalingResult,
    IsotonicScalingResult,
)


@dataclass
class CalibrationCfg:
    """Base config for post-hoc calibration. Specifies which method to use.

    Fitted parameters are stored back on the cfg (compare=False) after
    fitting and saved to config.json alongside the rest of the experiment.
    """
    calibration_type: str = field(default='none', init=False)
    name: str = 'None'

    def to_result(self) -> CalibrationResult | None:
        return None


@dataclass
class TemperatureScalingCfg(CalibrationCfg):
    """Single scalar temperature: softmax(logits / T)."""
    calibration_type: str = field(default='temperature', init=False)
    name: str = 'Temperature Scaling'
    T: float | None = field(default=None, compare=False)

    def to_result(self) -> TemperatureScalingResult | None:
        if self.T is None:
            return None
        return TemperatureScalingResult(T=self.T)


@dataclass
class VectorScalingCfg(CalibrationCfg):
    """Per-bin affine scaling: softmax(logits * w + b). Can shift argmax."""
    calibration_type: str = field(default='vector', init=False)
    name: str = 'Vector Scaling'
    w: list[float] | None = field(default=None, compare=False)
    b: list[float] | None = field(default=None, compare=False)

    def to_result(self) -> VectorScalingResult | None:
        if self.w is None or self.b is None:
            return None
        return VectorScalingResult(w=np.array(self.w), b=np.array(self.b))


@dataclass
class PlattScalingCfg(CalibrationCfg):
    """Affine transform on binary logit: sigmoid(a * logit + b). Binary only."""
    calibration_type: str = field(default='platt', init=False)
    name: str = 'Platt Scaling'
    a: float | None = field(default=None, compare=False)
    b: float | None = field(default=None, compare=False)

    def to_result(self) -> PlattScalingResult | None:
        if self.a is None or self.b is None:
            return None
        return PlattScalingResult(a=self.a, b=self.b)


@dataclass
class IsotonicScalingCfg(CalibrationCfg):
    """Monotone non-parametric calibration on the binary probability. Binary only.

    Fits a free step function from the model's predicted probability to the
    empirical event rate (sklearn IsotonicRegression). The fitted interpolation
    knots are stored so the map is reconstructed at load time without sklearn.
    """
    calibration_type: str = field(default='isotonic', init=False)
    name: str = 'Isotonic Scaling'
    x_thresholds: list[float] | None = field(default=None, compare=False)
    y_thresholds: list[float] | None = field(default=None, compare=False)

    def to_result(self) -> IsotonicScalingResult | None:
        if self.x_thresholds is None or self.y_thresholds is None:
            return None
        return IsotonicScalingResult(
            x_thresholds=np.array(self.x_thresholds),
            y_thresholds=np.array(self.y_thresholds),
        )


CALIBRATION: dict[str, CalibrationCfg | None] = {
    'None': CalibrationCfg(),
    'Temperature Scaling': TemperatureScalingCfg(),
    'Vector Scaling': VectorScalingCfg(),
    'Platt Scaling': PlattScalingCfg(),
    'Isotonic Scaling': IsotonicScalingCfg(),
}

CALIBRATION_TYPES: dict[str, type[CalibrationCfg]] = {
    'none': CalibrationCfg,
    'temperature': TemperatureScalingCfg,
    'vector': VectorScalingCfg,
    'platt': PlattScalingCfg,
    'isotonic': IsotonicScalingCfg,
}
