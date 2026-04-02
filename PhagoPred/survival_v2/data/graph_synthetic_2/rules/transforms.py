from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):

    def __init__(self, window: int) -> np.ndarray:
        self.window = window

    def __call__(self, f: np.ndarray) -> np.ndarray:
        if len(f) < self.window:
            window = len(f)
            if len(f) == 0:
                return np.zeros(1)
        else:
            window = self.window
        f = f[-window:]
        result = self._compute(f)
        return result
        # return self._apply_window(result)

    @abstractmethod
    def _compute(self, f: np.ndarray) -> np.ndarray:
        ...


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


class Threshold(Transform):

    def __init__(self, window: int, threshold: float, binary: bool):
        super().__init__(window)
        self.threshold = threshold
        self.binary = binary

    def _compute(self, f: np.ndarray) -> np.ndarray:
        mask = f > self.threshold
        if self.binary:
            return mask
        else:
            f[~mask] = 0
            return f


class LocalVariance(Transform):

    def _compute(self, f: np.ndarray) -> np.ndarray:
        variance = np.array(
            [np.var(f[max(0, t - self.window):t + 1]) for t in range(len(f))])
        return variance

    # def __call__(self, f: np.ndarray) -> np.ndarray:
    #     return self._compute(f)


TRANSFORMS = [
    Value,
    Absolute,
    Gradient,
    SecondDerivative,
    Cumulative,
]
