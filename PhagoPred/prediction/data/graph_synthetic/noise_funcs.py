from abc import ABC, abstractmethod

import numpy as np


class Noise(ABC):

    @abstractmethod
    def sample(self, T: int) -> np.ndarray:
        ...

    @abstractmethod
    def log_prob(self, val: float) -> float:
        ...


class GaussianNoise(Noise):

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def sample(self, T: int) -> np.ndarray:
        return np.random.randn(T) * self.sigma

    def log_prob(self, val: float) -> float:
        return -0.5 * (
            (val / self.sigma)**2 + np.log(2 * np.pi * self.sigma**2))


class LaplaceNoise(Noise):

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def sample(self, T: int) -> np.ndarray:
        return np.random.laplace(0, self.scale, T)

    def log_prob(self, val):
        pass


class NoNoise(Noise):

    def sample(self, T: int) -> np.ndarray:
        return np.zeros(T)

    def log_prob(self, val):
        pass
