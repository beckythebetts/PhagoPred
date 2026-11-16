from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


def _gaussian_sample(T, rng=None, sigma=1.0):
    return (rng or np.random).normal(0.0, sigma, T)


def _gaussian_log_prob(x, sigma=1.0):
    return -0.5 * ((x / sigma)**2 + np.log(2 * np.pi * sigma**2))


def _laplace_sample(T, rng=None, scale=1.0):
    return (rng or np.random).laplace(0.0, scale, T)


def _laplace_log_prob(x, scale=1.0):
    return -np.abs(x) / scale - np.log(2 * scale)


def _none_sample(T, rng=None):
    return np.zeros(T)


def _none_log_prob(x):
    return np.zeros_like(x, dtype=float)


_DISTRIBUTIONS = {
    'gaussian': (_gaussian_sample, _gaussian_log_prob),
    'laplace': (_laplace_sample, _laplace_log_prob),
    'none': (_none_sample, _none_log_prob),
}


@dataclass
class Noise:
    """A noise distribution: which kind, and its parameters."""
    kind: str = 'none'
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _DISTRIBUTIONS:
            raise ValueError(
                f'unknown noise kind {self.kind!r}; known: {list(_DISTRIBUTIONS)}'
            )
        self.sample(1)

    def sample(self, T: int, rng=None) -> np.ndarray:
        return _DISTRIBUTIONS[self.kind][0](T, rng, **self.params)

    def log_prob(self, x) -> np.ndarray:
        return _DISTRIBUTIONS[self.kind][1](x, **self.params)

    @classmethod
    def gaussian(cls, sigma: float) -> Noise:
        return cls('gaussian', {'sigma': sigma})

    @classmethod
    def laplace(cls, scale: float) -> Noise:
        return cls('laplace', {'scale': scale})

    @classmethod
    def none(cls) -> Noise:
        return cls('none')
