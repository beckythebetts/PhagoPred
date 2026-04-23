"""
Axis 2 — Function Family
========================
Each class maps a single scalar value to a scalar output.
Used as the per-parent component in a StructuralForm.
"""
from abc import ABC, abstractmethod
import numpy as np


class ComponentFunc(ABC):
    """Maps a single parent value x -> scalar."""

    def __call__(self, x: float) -> float:
        return self._compute(float(x))

    @abstractmethod
    def _compute(self, x: float) -> float: ...


# ── Linear ────────────────────────────────────────────────────────────────────

class Linear(ComponentFunc):
    """f(x) = slope * x + intercept"""

    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        self.slope = slope
        self.intercept = intercept

    def _compute(self, x):
        return self.slope * x + self.intercept


# ── Monotone smooth ───────────────────────────────────────────────────────────

class Hill(ComponentFunc):
    """Hill activation: x^n / (k^n + x^n).

    Ranges 0→1. k = half-saturation constant, n = cooperativity (steepness).
    """

    def __init__(self, k: float = 1.0, n: float = 1.0):
        self.k = k
        self.n = n

    def _compute(self, x):
        if x <= 0:
            return 0.0
        xn = x ** self.n
        return xn / (self.k ** self.n + xn)


class Sigmoid(ComponentFunc):
    """Logistic sigmoid: 1 / (1 + exp(-k * (x - x0))).

    x0 = inflection point, k = steepness.
    """

    def __init__(self, k: float = 1.0, x0: float = 0.0):
        self.k = k
        self.x0 = x0

    def _compute(self, x):
        return 1.0 / (1.0 + np.exp(-self.k * (x - self.x0)))


class Power(ComponentFunc):
    """f(x) = sign(x) * |x|^exponent."""

    def __init__(self, exponent: float = 2.0):
        self.exponent = exponent

    def _compute(self, x):
        return np.sign(x) * (abs(x) ** self.exponent)


# ── Threshold / piecewise ─────────────────────────────────────────────────────

class Threshold(ComponentFunc):
    """Dead-zone then linear: 0 if x < theta, else scale * (x - theta)."""

    def __init__(self, theta: float = 0.0, scale: float = 1.0):
        self.theta = theta
        self.scale = scale

    def _compute(self, x):
        return self.scale * max(0.0, x - self.theta)


class Piecewise(ComponentFunc):
    """Arbitrary piecewise-linear function defined by (x, y) breakpoints.

    Values outside the range are clamped to the nearest endpoint.
    """

    def __init__(self, xs: list[float], ys: list[float]):
        if len(xs) != len(ys) or len(xs) < 2:
            raise ValueError("xs and ys must be equal length >= 2")
        self.xs = np.array(xs, dtype=float)
        self.ys = np.array(ys, dtype=float)

    def _compute(self, x):
        return float(np.interp(x, self.xs, self.ys))


# ── Periodic / oscillatory ────────────────────────────────────────────────────

class Sine(ComponentFunc):
    """f(x) = amplitude * sin(frequency * x + phase)."""

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def _compute(self, x):
        return self.amplitude * np.sin(self.frequency * x + self.phase)


class Cosine(ComponentFunc):
    """f(x) = amplitude * cos(frequency * x + phase)."""

    def __init__(self, amplitude: float = 1.0, frequency: float = 1.0, phase: float = 0.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def _compute(self, x):
        return self.amplitude * np.cos(self.frequency * x + self.phase)


# ── Non-monotone smooth ───────────────────────────────────────────────────────

class Polynomial(ComponentFunc):
    """f(x) = sum_i coefficients[i] * x^i  (ascending powers).

    E.g. Polynomial([0, 0, 1]) = x^2.
    """

    def __init__(self, coefficients: list[float]):
        self.coefficients = np.array(coefficients, dtype=float)

    def _compute(self, x):
        return float(np.polyval(self.coefficients[::-1], x))


class GaussianBump(ComponentFunc):
    """f(x) = amplitude * exp(-0.5 * ((x - centre) / width)^2).

    Non-monotone, peaks at centre. Models optimal-range effects.
    """

    def __init__(self, centre: float = 0.0, width: float = 1.0, amplitude: float = 1.0):
        self.centre = centre
        self.width = width
        self.amplitude = amplitude

    def _compute(self, x):
        return self.amplitude * np.exp(-0.5 * ((x - self.centre) / self.width) ** 2)
