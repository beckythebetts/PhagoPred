from abc import ABC, abstractmethod
import numpy as np


class Magnitude(ABC):
    """
    Abstract base class for scaling a shaped trigger into a hazard contribution.

    A Magnitude operation scales the output of the Shape stage before it
    is added to the hazard. This is the final stage of the rule pipeline:
        feature → Transform → Exceedance → Combination → Timing → Shape → Magnitude → hazard
    """

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return self._compute(signal)

    @abstractmethod
    def _compute(self, signal: np.ndarray) -> np.ndarray: ...


class Fixed(Magnitude):
    """
    Scale the signal by a fixed constant.

    Args:
        scale (float): Multiplicative scaling factor applied to the signal.
    """

    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        return signal * self.scale


class Saturating(Magnitude):
    """
    Scale by `scale` then clip at `maximum`.

    Models diminishing returns — the hazard contribution cannot exceed a
    ceiling regardless of how large the trigger signal becomes.

    Args:
        scale (float): Multiplicative scaling factor.
        maximum (float): Upper bound on the hazard contribution.
    """

    def __init__(self, scale: float = 0.01, maximum: float = 0.1):
        self.scale = scale
        self.maximum = maximum

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        return np.clip(signal * self.scale, 0, self.maximum)


class RandomFixed(Magnitude):
    """
    Scale by a value drawn uniformly from [min_scale, max_scale].

    The scale is sampled once per rule application, giving each cell a
    different effect strength. Models biological variability in sensitivity.

    Args:
        min_scale (float): Lower bound of the uniform scale distribution.
        max_scale (float): Upper bound of the uniform scale distribution.
    """

    def __init__(self, min_scale: float = 0.001, max_scale: float = 0.05):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def _compute(self, signal: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return signal * scale