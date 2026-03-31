from abc import ABC, abstractmethod
import numpy as np


class Threshold(ABC):

    @abstractmethod
    def __init__(self, threshold: float):
        self.threshold = threshold

    @abstractmethod
    def __call__(self, f: np.ndarray) -> np.ndarray:
        ...


class Instantaneous(Threshold):

    def _call__(self, f: np.ndarray) -> np.ndarray:
        return np.clip(f, a_min=self.threshold, a_max=None)


class Persistence(Threshold):

    def __call__(self, f: np.ndarray) -> np.ndarray:
        above = (f > self.threshold).astype(int)
        count = np.zeros(len(f))
        for t in range(len(f)):
            if above[t]:
                count[t] = count[t - 1] + 1 if t > 0 else 1
            else:
                count[t] = 0
        return count


class Cumulative(Threshold):

    def __call__(self, f: np.ndarray) -> np.ndarray:
        above = (f > self.threshold).astype(int)
        return np.cumsum(above)


ALL_THRESHOLDS = [Instantaneous, Persistence, Cumulative]
