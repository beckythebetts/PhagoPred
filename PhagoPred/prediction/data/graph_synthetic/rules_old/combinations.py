from abc import ABC, abstractmethod
import numpy as np


class Combination(ABC):
    """
    Abstract base class for combining multiple trigger signals into one.

    Takes a list of arrays of shape (T,) and reduces them to a single
    array of shape (T,). The combination operation determines how multiple
    conditions interact to produce the final trigger signal passed to
    Timing and Shape stages.
    """

    def __call__(self, signals: list[np.ndarray]) -> np.ndarray:
        min_signal_length = min(len(sig) for sig in signals)
        signals = [sig[:min_signal_length] for sig in signals]
        return self._compute(signals)

    @abstractmethod
    def _compute(self, signals: list[np.ndarray]) -> np.ndarray:
        ...


class Add(Combination):
    """Sum all signals — each condition contributes independently."""

    def _compute(self, signals):
        return np.sum(signals, axis=0)


class Multiply(Combination):
    """
    Product of all signals — all conditions must be simultaneously active.
    Equivalent to a quantitative AND: output is zero if any signal is zero.
    """

    def _compute(self, signals):
        return np.prod(signals, axis=0)


class Max(Combination):
    """Maximum across signals — dominated by the strongest condition."""

    def _compute(self, signals):
        return np.max(signals, axis=0)


class Min(Combination):
    """
    Minimum across signals — bottlenecked by the weakest condition.
    Equivalent to a quantitative AND with saturation at the weakest input.
    """

    def _compute(self, signals):
        return np.min(signals, axis=0)


class Mean(Combination):
    """Mean of all signals — equal contribution from each condition."""

    def _compute(self, signals):
        return np.mean(signals, axis=0)
