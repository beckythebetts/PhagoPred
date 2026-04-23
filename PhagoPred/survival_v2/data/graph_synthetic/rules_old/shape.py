from abc import ABC, abstractmethod
import numpy as np


class Shape(ABC):
    """
    Abstract base class for shaping a timed trigger into a hazard contribution.

    A Shape operation convolves or transforms a trigger signal into a
    smoothed hazard contribution. This controls how the effect of a rule
    spreads over time — whether it is a sharp spike, a smooth bump, a
    sustained plateau, or a permanent step.
    """

    def __call__(self, f: np.ndarray, t: int, value: float) -> np.ndarray:
        return self._compute(f, t, value)

    @abstractmethod
    def _compute(self, f: np.ndarray, t: int, value: float):
        ...


class Delta(Shape):

    def _compute(self, f: np.ndarray, t: int, value: float):
        if t < len(f):
            f[t] += value
        return f


class Gaussian(Shape):

    def __init__(self, std: float):
        self.std = std

    def _compute(self, f: np.ndarray, t: int, value: float):
        T = len(f)
        radius = int(3 * self.std)
        indices = np.arange(max(0, t - radius), min(T, t + radius + 1))
        offsets = indices - t
        weights = np.exp(-0.5 * (offsets / self.std)**2)
        # weights /= weights.sum()  # normalise so contributions sum to value
        f[indices] += value * weights
        return f


class ExponentialDecay(Shape):
    """Sharp onset at t, then exponential decay.

    Models signal termination processes — receptor desensitisation,
    second-messenger clearance, drug washout. The most common waveform
    in intracellular signalling.

    Args:
        rate: decay rate constant (1/timescale). Higher = faster decay.
    """

    def __init__(self, rate: float):
        self.rate = rate

    def _compute(self, f: np.ndarray, t: int, value: float):
        T = len(f)
        horizon = int(6 / self.rate)  # captures ~99.75% of mass
        indices = np.arange(t, min(T, t + horizon + 1))
        offsets = indices - t
        weights = np.exp(-self.rate * offsets)
        # weights /= weights.sum()
        f[indices] += value * weights
        return f


class Gamma(Shape):
    """Delayed rise to a peak then gradual decay.

    Models processes that require a build-up phase before peaking —
    cytokine release, inflammatory infiltration, adaptive immune responses.
    The peak occurs at time t + (shape-1)*scale after the trigger.

    Args:
        shape: controls steepness of rise (k >= 1; k=1 is pure exponential decay).
        scale: timescale (theta). Peak at offset (k-1)*theta from t.
    """

    def __init__(self, shape: float = 2.0, scale: float = 5.0):
        if shape < 1:
            raise ValueError("shape must be >= 1")
        self.shape = shape
        self.scale = scale

    def _compute(self, f: np.ndarray, t: int, value: float):
        T = len(f)
        horizon = int((self.shape + 5) * self.scale)  # well past the tail
        indices = np.arange(t, min(T, t + horizon + 1))
        offsets = (indices - t).astype(float)
        # avoid 0^(k-1) issue when shape > 1
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(
                offsets == 0,
                0.0 if self.shape > 1 else 1.0,
                (offsets / self.scale)**(self.shape - 1) *
                np.exp(-offsets / self.scale),
            )
        # weights /= weights.sum()
        f[indices] += value * weights
        return f


class BiExponential(Shape):
    """Fast rise, slower decay — the canonical synaptic/receptor waveform.

    Kernel: exp(-t/tau_decay) - exp(-t/tau_rise), normalised.
    Models ligand-receptor binding kinetics, EPSP shapes, and any process
    with distinct association and dissociation time constants.

    Args:
        tau_rise: rise time constant. Must be < tau_decay.
        tau_decay: decay time constant.
    """

    def __init__(self, tau_rise: float = 2.0, tau_decay: float = 10.0):
        if tau_rise >= tau_decay:
            raise ValueError("tau_rise must be < tau_decay")
        self.tau_rise = tau_rise
        self.tau_decay = tau_decay

    def _compute(self, f: np.ndarray, t: int, value: float):
        T = len(f)
        horizon = int(6 * self.tau_decay)
        indices = np.arange(t, min(T, t + horizon + 1))
        offsets = (indices - t).astype(float)
        weights = np.exp(-offsets / self.tau_decay) - np.exp(
            -offsets / self.tau_rise)
        weights = np.clip(weights, 0, None)
        if weights.sum() == 0:
            return f
        # weights /= weights.sum()
        f[indices] += value * weights
        return f


class Plateau(Shape):
    """Sustained flat effect for a fixed duration, then abrupt termination.

    Models refractory periods, sustained gene expression windows,
    or any binary on/off biological switch that remains active for
    a set number of timesteps after triggering.

    Args:
        duration: number of timesteps the effect is sustained.
    """

    def __init__(self, duration: int):
        self.duration = duration

    def _compute(self, f: np.ndarray, t: int, value: float):
        T = len(f)
        indices = np.arange(t, min(T, t + self.duration))
        f[indices] += value / len(indices)
        return f
