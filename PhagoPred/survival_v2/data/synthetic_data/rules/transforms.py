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
        self._compute(f)


TRANSFORMS = [
    Value,
    Absolute,
    Gradient,
    SecondDerivative,
    Cumulative,
]

# class Transform(Enum):
#     """Define functions for searching for Transforms in given freature"""
#     THRESHOLD = auto()
#     VALUE = auto()
#     ABS = auto()
#     GRADIENT = auto()
#     SECOND_DERIVATIVE = auto()
#     PERSISTENCE = auto()

#     # def __call__(self, f: np.ndarray, **params) -> np.ndarray:
#     #     return getattr(self, f'_{self.name.lower()}')(f, **params)
#     def __call__(self, **params):
#         """Returns a configured trigger function."""
#         def trigger(f: np.ndarray) -> np.ndarray:
#             return getattr(self, f'_{self.name.lower()}')(f, **params)
#         return trigger

#     def _threshold(self, f: np.ndarray, threshold: float, **kw) -> np.ndarray:
#         """Return f - threshold where f>threshold"""
#         return np.clip(f, a_min=threshold)

#     def _value(self, f: np.ndarray) -> np.ndarray:
#         return f

#     def _abs(self, f: np.ndarray) -> np.ndarray:
#         return abs(f)

#     def _gradient(self, f: np.ndarray, window: int) -> np.ndarray:
#         grad = np.diff(f, prepend=f[0])
#         if window > 1:
#             grad = np.convolve(grad, np.ones(window) / window, mode='same')
#         return grad

#     def _second_derivative(self, f: np.ndarray, window: int) -> np.ndarray:
#         grad = np.diff(f, prepend=f[0])
#         second_derviative = np.diff(grad, prepend=grad[0])
#         if window > 1:
#             second_derviative = np.convolve(second_derviative,
#                                             np.ones(window) / window,
#                                             mode='same')
#         return

#     def persistence(self)
