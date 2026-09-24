from dataclasses import dataclass, field

import numpy as np

from .transforms import Transform
from .combinations import Combination, Add
from .collapse import Collapse

from .timing import Timing, Fixed
from .shape import Shape
from .magnitude import Magnitude
from ..structural_forms import StructuralForm


@dataclass
class Input:
    feature: str
    transform: Transform


@dataclass
class Rule:
    inputs: list[Input]
    target: str
    collapse: Collapse
    magnitude: Magnitude
    timing: Timing
    shape: Shape
    combination: Combination = Add()

    @property
    def max_history(self):
        return max(inp.transform.window for inp in self.inputs)

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> np.ndarray:
        # signals = signals.copy()
        if t <= self.max_history:
            return signals
        triggers = []
        for inp in self.inputs:
            f = signals[inp.feature][:t].copy()
            trigger = inp.transform(f)
            triggers.append(trigger)
        combined = self.combination(triggers)
        value = self.collapse(combined)
        value = self.magnitude(value)
        delay = self.timing()
        effect = self.shape(
            signals[self.target][t:].copy(),
            delay,
            value,
        )
        signals[self.target][t:] = effect
        return signals


@dataclass
class AccumulatorRule:
    """
    AR(1) accumulator with external driving.

    A[t] = A[t-1] + magnitude(driver(t-1))

    The target carries its own previous value forward at every step,
    plus an increment from the driving inputs. This produces a random
    walk with drift proportional to the driving signal.
    """
    inputs: list[Input]
    target: str
    collapse: Collapse
    magnitude: Magnitude
    combination: Combination = Add()
    timing: Timing = Fixed(delay=0)

    @property
    def max_history(self) -> int:
        return max(inp.transform.window for inp in self.inputs)

    def _compute_driver(self, signals: dict, t: int) -> float:
        triggers = []
        for inp in self.inputs:
            f = signals[inp.feature][:t].copy()
            triggers.append(inp.transform(f))
        combined = self.combination(triggers)
        return float(self.collapse(combined))

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> dict:
        if t <= self.max_history:
            return signals
        carry = signals[self.target][t - 1]
        increment = self.magnitude(self._compute_driver(signals, t))
        signals[self.target][t] += carry + increment
        return signals


@dataclass
class DecayingAccumulatorRule(AccumulatorRule):
    """
    Accumulates while the driver exceeds a threshold, decays otherwise.

    Above threshold : A[t] = A[t-1] + magnitude(driver(t-1))
    Below threshold : A[t] = A[t-1] * decay_rate

    Models processes that build up under stimulation but return toward
    baseline when the stimulus is absent — e.g. a stress/damage signal
    that accumulates while a threat persists and slowly resolves when it
    is removed. decay_rate < 1 guarantees stationarity during the off phase.
    """
    threshold: float = 0.0
    decay_rate: float = 0.9

    def __post_init__(self):
        if not 0 < self.decay_rate < 1:
            raise ValueError(
                f"decay_rate must be in (0, 1), got {self.decay_rate}")

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> dict:
        if t <= self.max_history:
            return signals
        carry = signals[self.target][t - 1]
        driver = self._compute_driver(signals, t)
        if driver > self.threshold:
            signals[self.target][t] += carry + self.magnitude(driver)
        else:
            signals[self.target][t] += carry * self.decay_rate
        return signals


@dataclass
class SEMRule:
    """Structural Equation Model rule for fixed-lag, window=1, delta effects.

    Implements: target[t] += structural_form({parent: parent[t - lag]})

    The causal lag is explicit and exact — no off-by-one ambiguity.
    Parents are always read at t - lag (a single past value, window=1).
    The increment is added as a delta at t.

    Args:
        inputs:           list of parent feature names.
        target:           name of the target feature.
        structural_form:  StructuralForm instance (defines Axis 1 × Axis 2).
        lag:              causal delay in timesteps (must be >= 1).
    """
    inputs: list[str]
    target: str
    structural_form: StructuralForm
    lag: int = 1

    def __post_init__(self):
        if self.lag < 1:
            raise ValueError(f"lag must be >= 1, got {self.lag}")

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> dict:
        if t < self.lag:
            return signals
        parent_values = {
            name: signals[name][t - self.lag]
            for name in self.inputs
        }
        signals[self.target][t] += self.structural_form(parent_values)
        return signals
