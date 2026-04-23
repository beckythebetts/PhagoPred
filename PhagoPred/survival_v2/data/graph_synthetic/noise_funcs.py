from abc import ABC, abstractmethod

import numpy as np


class Noise(ABC):

    @abstractmethod
    def sample(self, T: int) -> np.ndarray:
        ...


class GaussianNoise(Noise):

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def sample(self, T: int) -> np.ndarray:
        return np.random.randn(T) * self.sigma


class LaplaceNoise(Noise):

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def sample(self, T: int) -> np.ndarray:
        return np.random.laplace(0, self.scale, T)


class NoNoise(Noise):

    def sample(self, T: int) -> np.ndarray:
        return np.zeros(T)
