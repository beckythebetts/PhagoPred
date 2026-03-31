from abc import ABC, abstractmethod
import numpy as np
from scipy.ndimage import gaussian_filter1d


class Shape(ABC):
    """
    Abstract base class for shaping a timed trigger into a hazard contribution.

    A Shape operation convolves or transforms a trigger signal into a
    smoothed hazard contribution. This controls how the effect of a rule
    spreads over time — whether it is a sharp spike, a smooth bump, a
    sustained plateau, or a permanent step.
    """

    def __call__(self, trigger: np.ndarray) -> np.ndarray:
        return self._compute(trigger)

    @abstractmethod
    def _compute(self, trigger: np.ndarray) -> np.ndarray: ...


class Delta(Shape):
    """
    No shaping — the trigger signal is passed through unchanged.

    The hazard contribution is non-zero only at frames where the trigger
    is non-zero. Use when the effect is instantaneous and localised.
    """

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        return trigger


class Step(Shape):
    """
    Permanent step — once the trigger fires, the hazard stays elevated.

    The output is the cumulative maximum of the trigger, so any non-zero
    trigger frame causes a sustained increase for all subsequent frames.
    Models irreversible damage or state changes.
    """

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(trigger)


class Gaussian(Shape):
    """
    Smooth the trigger with a Gaussian kernel.

    Spreads each trigger spike into a smooth bump of width `sigma` frames.
    Models effects that build up and decay gradually around the trigger time.

    Args:
        sigma (float): Standard deviation of the Gaussian kernel in frames.
    """

    def __init__(self, sigma: float = 20.0):
        self.sigma = sigma

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(trigger, sigma=self.sigma)


class Ramp(Shape):
    """
    Linear ramp up over `ramp_length` frames from each trigger onset.

    At each frame where the trigger becomes non-zero after a zero, a linear
    ramp of length `ramp_length` is added to the output. Models gradual
    accumulation of effect after a condition activates.

    Args:
        ramp_length (int): Number of frames over which the effect ramps up.
    """

    def __init__(self, ramp_length: int = 50):
        self.ramp_length = ramp_length

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        output = np.zeros_like(trigger)
        T = len(trigger)
        for t in range(T):
            if trigger[t] > 0:
                end = min(T, t + self.ramp_length)
                ramp = np.linspace(0, trigger[t], end - t)
                output[t:end] += ramp
        return output


class Plateau(Shape):
    """
    Sustain the trigger signal for `duration` frames after each onset.

    Each non-zero trigger frame is extended forward by `duration` frames
    at the same amplitude. Models effects that persist for a fixed window
    after a condition activates, then cease abruptly.

    Args:
        duration (int): Number of frames to sustain each trigger activation.
    """

    def __init__(self, duration: int = 50):
        self.duration = duration

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        output = np.zeros_like(trigger)
        T = len(trigger)
        for t in range(T):
            if trigger[t] > 0:
                end = min(T, t + self.duration)
                output[t:end] = np.maximum(output[t:end], trigger[t])
        return output


class ExponentialDecay(Shape):
    """
    Sharp rise at the trigger followed by exponential decay.

    At each non-zero trigger frame, adds an exponentially decaying
    contribution of the form trigger[t] * exp(-dt / tau) for subsequent
    frames. Models impulse responses — a rapid effect that fades over time.

    Args:
        tau (float): Decay time constant in frames. Larger values give
            slower decay and longer-lasting effects.
    """

    def __init__(self, tau: float = 30.0):
        self.tau = tau

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        output = np.zeros_like(trigger)
        T = len(trigger)
        for t in range(T):
            if trigger[t] > 0:
                dt = np.arange(T - t)
                output[t:] += trigger[t] * np.exp(-dt / self.tau)
        return output