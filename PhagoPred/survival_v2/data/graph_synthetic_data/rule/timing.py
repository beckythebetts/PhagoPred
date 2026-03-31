from abc import ABC, abstractmethod
import numpy as np


class Timing(ABC):
    """
    Abstract base class for temporal shifting of a trigger signal.

    A Timing operation shifts or spreads a trigger signal in time before
    it is passed to the Shape stage. This models the lag between a causal
    condition becoming active and its effect on the hazard.
    """

    def __call__(self, trigger: np.ndarray) -> np.ndarray:
        return self._compute(trigger)

    def apply_step(self, signal: np.ndarray, t: int, value: float) -> float:
        """
        Return the contribution at timestep t for autoregressive rules.

        Default implementation is instantaneous — returns value unchanged.
        Subclasses override this to read from past timesteps instead.

        Args:
            signal: the target feature's trajectory, partially filled up to t.
            t: current timestep.
            value: trigger value computed at timestep t.

        Returns:
            float: contribution to the target at timestep t.
        """
        return value

    @abstractmethod
    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        ...


class Instantaneous(Timing):
    """No shift — hazard effect occurs at the same frame as the trigger."""

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        return trigger


class FixedDelay(Timing):
    """
    Shift the trigger signal forward by a fixed number of frames.

    Models a deterministic lag between cause and effect — e.g. a cellular
    stress signal that takes `delay` frames to propagate to a death outcome.

    Args:
        delay (int): Number of frames to shift the signal forward.
    """

    def __init__(self, delay: int = 10):
        self.delay = delay

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        shifted = np.zeros_like(trigger)
        shifted[self.delay:] = trigger[:len(trigger) - self.delay]
        return shifted

    def apply_step(self, signal: np.ndarray, t: int, value: float) -> float:
        t_source = t - self.delay
        return float(signal[t_source]) if t_source >= 0 else 0.0


class VariableDelay(Timing):
    """
    Shift the trigger signal forward by a delay drawn from N(mean, sigma).

    Models stochastic lag — the delay varies across cells, simulating
    biological variability in response time.

    Args:
        mean (int): Mean delay in frames.
        sigma (float): Standard deviation of the delay distribution.
    """

    def __init__(self, mean: int = 50, sigma: float = 10.0):
        self.mean = mean
        self.sigma = sigma

    def _compute(self, trigger: np.ndarray) -> np.ndarray:
        delay = max(0, int(np.random.normal(self.mean, self.sigma)))
        shifted = np.zeros_like(trigger)
        shifted[delay:] = trigger[:len(trigger) - delay]
        return shifted
