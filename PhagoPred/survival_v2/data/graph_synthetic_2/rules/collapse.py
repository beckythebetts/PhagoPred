from abc import ABC, abstractmethod

import numpy as np


class Collapse(ABC):
    """Classes for applying different functions to collapse time sereis variable to single value"""

    @abstractmethod
    def __call__(self, a: np.ndarray) -> float:
        ...


class Max(Collapse):

    def __call__(self, a: np.ndarray) -> float:
        return max(a)


class Mean(Collapse):

    def __call__(self, a: np.ndarray) -> float:
        return np.mean(a)


class Range(Collapse):

    def __call__(self, a: np.ndarray) -> float:
        return max(a) - min(a)


class WeightedMean(Collapse):

    def __init__(self, weights: np.ndarray) -> float:
        self.weights = weights

    def __call__(self, a: np.ndarray) -> float:
        return np.sum(a * self.weights) / np.sum(self.weights)


class ExponentialDecayWeightedMean(WeightedMean):
    """Weighted mean with exponential recency bias.

    For a window [a_0, ..., a_{N-1}] where a_{N-1} is the most recent frame,
    weight for a_i = alpha^(N-1-i), so the most recent value has weight 1
    and each older step is discounted by alpha.
    """

    def __init__(self, alpha: float = 0.9):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def __call__(self, a: np.ndarray) -> float:
        n = len(a)
        self.weights = self.alpha**np.arange(n - 1, -1, -1)
        return super().__call__(a)
