from abc import ABC, abstractmethod

import numpy as np


class BaseFunc(ABC):

    @abstractmethod
    def generate(self, T: int) -> np.ndarray:
        ...


class Constant(BaseFunc):

    def __init__(self, value: float = 0.0):
        self.value = value

    def generate(self, T):
        return np.full(T, self.value)


class Ramp(BaseFunc):

    def __init__(self, slope: float = 1.0):
        self.slope = slope

    def generate(self, T):
        return np.linspace(0, self.slope * T, T)


class Oscillation(BaseFunc):

    def __init__(self, amplitude, frequency, phase=0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def generate(self, T):
        return self.amplitude * np.sin(2 * np.pi * self.frequency *
                                       np.arange(T) + self.phase)


class Polynomial(BaseFunc):

    def __init__(self, coeffs: list[float]):
        self.coeffs = coeffs

    def generate(self, T):
        return np.polyval(self.coeffs, np.arange(T))


class RandomWalk(BaseFunc):

    def __init__(self, step_scale: float = 1.0):
        self.step_scale = step_scale

    def generate(self, T):
        steps = np.random.randn(T) * self.step_scale
        return np.cumsum(steps)
