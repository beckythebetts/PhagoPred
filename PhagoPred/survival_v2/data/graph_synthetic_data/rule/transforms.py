# from abc import ABC, abstractmethod
# import numpy as np

# class Transform(ABC):
#     """
#     Abstract base class for feature transforms.

#     A Transform maps a raw feature trajectory f of shape (T,) to a derived
#     signal of the same shape, summarising a different property of the signal
#     over a rolling window of `window` past frames.

#     Subclasses implement `_compute`. The `window` parameter controls how many
#     past frames are included in the computation at each timestep — window=1
#     means only the current and immediately preceding frame are used.
#     """

#     def __init__(self, window: int):
#         self.window = window

#     def __call__(self, f: np.ndarray) -> np.ndarray:
#         return self._compute(f)

#     @abstractmethod
#     def _compute(self, f: np.ndarray) -> np.ndarray:
#         ...

# class Value(Transform):
#     def _compute(self, f: np.ndarray) -> np.ndarray:

# class Gradient(Transform):
#     """
#     Compute the average slope of f over the last `window` frames.

#     At each timestep t, returns (f[t] - f[t - window]) / window — the mean
#     rate of change over the lookback window. Returns 0.0 at frames where
#     fewer than `window` past frames are available.

#     Args:
#         window (int): Number of past frames to include. window=1 gives the
#             instantaneous frame-to-frame difference.
#     """

#     def _compute(self, f: np.ndarray) -> np.ndarray:
#         grad = np.zeros_like(f)
#         for t in range(len(f)):
#             start = max(0, t - self.window)
#             grad[t] = (f[t] - f[start]) / (t - start) if t > start else 0.0
#         return grad

# class Acceleration(Transform):
#     """
#     Compute the average rate of change of the gradient over the last `window` frames.

#     First computes a rolling gradient (as in `Gradient`), then applies the
#     same rolling slope to the gradient signal. This approximates the second
#     derivative of f — how quickly the rate of change is itself changing.
#     Returns 0.0 at frames where fewer than `window` past frames are available.

#     Args:
#         window (int): Lookback window applied to both gradient and acceleration
#             computations.
#     """

#     def _compute(self, f: np.ndarray) -> np.ndarray:
#         grad = np.zeros_like(f)
#         accel = np.zeros_like(f)
#         for t in range(len(f)):
#             start = max(0, t - self.window)
#             grad[t] = (f[t] - f[start]) / (t - start) if t > start else 0.0
#             accel[t] = (grad[t] - grad[start]) / (t -
#                                                   start) if t > start else 0.0
#         return accel

# class LocalVariance(Transform):
#     """
#     Compute the variance of f over a rolling window of `window` past frames.

#     At each timestep t, returns the variance of f[max(0, t-window) : t+1].
#     This captures local volatility — how much the signal fluctuates in the
#     recent past, independent of its absolute value or trend.

#     Args:
#         window (int): Number of past frames included in the variance estimate.
#             Larger values give a smoother, longer-memory variance signal.
#     """

#     def _compute(self, f: np.ndarray) -> np.ndarray:
#         return np.array(
#             [np.var(f[max(0, t - self.window):t + 1]) for t in range(len(f))])
from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):

    def __init__(self, window: int) -> np.ndarray:
        self.window = window
        self.window_kernel = np.ones(self.window) / self.window

    def __call__(self, f: np.ndarray) -> np.ndarray:
        result = self._compute(f)
        return self._apply_window(result)

    @abstractmethod
    def _compute(self, f: np.ndarray) -> np.ndarray:
        ...

    def _apply_window(self, a: np.ndarray) -> np.ndarray:
        if self.window > 1:
            a = np.convolve(a, self.window_kernel, mode='same')
        return a


class Value(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        return f


class Absolute(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        return abs(f)


class Gradient(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        return np.diff(f, prepend=f[0])


class SecondDerivative(Transform):

    def _compute(self, f):
        grad = np.diff(f, prepend=f[0])
        return np.diff(grad, prepend=grad[0])


class Cumulative(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        return np.cumsum(f)


class LocalVariance(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        variance = np.array(
            [np.var(f[max(0, t - self.window):t + 1]) for t in range(len(f))])
        return variance

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return self._compute(f)


TRANSFORMS = [
    Value,
    Absolute,
    Gradient,
    SecondDerivative,
    Cumulative,
]
