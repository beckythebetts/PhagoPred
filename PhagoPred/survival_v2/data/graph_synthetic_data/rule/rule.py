from dataclasses import dataclass, field

import numpy as np

from .transforms import Transform
from .thresholds import Threshold
from .combinations import Combination
from .timing import Timing, FixedDelay, VariableDelay, Instantaneous
from .shape import Shape, Delta
from .magnitude import Magnitude


@dataclass
class Input:
    feature: str
    transform: Transform
    threshold: Threshold


@dataclass
class Rule:
    inputs: list[Input]
    target: str
    combination: Combination
    magnitude: Magnitude
    timing: Timing
    shape: Shape

    # def apply(self, signals: dict[str, np.ndarray]) -> np.ndarray:
    #     triggers = []
    #     for inp in self.inputs:
    #         f = signals[inp.feature]
    #         trasnformed = inp.transform(f)
    #         trigger = inp.threshold(trasnformed)
    #         triggers.append(trigger)

    #     combined = self.combination(triggers)

    #     timed = self.timing(combined)
    #     shaped = self.shape(timed)
    #     return self.magnitude(shaped)
    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> np.ndarray:
        triggers = []
        for inp in self.inputs:
            f = signals[inp.feature][:t]
            transformed = inp.transform(f)
            trigger = inp.threshold(transformed)
            triggers.append(trigger)
        combined = self.combination(triggers)
        timed =
    
    
    def get_delay(self) -> int:
        if isinstance(self.timing, FixedDelay):
            return self.timing.delay
        elif isinstance(self.timing, Instantaneous):
            return 0
        elif isinstance(self.timing, VariableDelay):
            return self.timing.mean
        else:
            raise ValueError(
                "Cannot determine delay for non-FixedDelay timing")


@dataclass
class AutoRegressiveRule(Rule):
    """
    A Rule restricted to step-by-step simulation for self-referencing features.

    Shape is fixed to Delta — autoregressive rules only write to the current
    timestep, never to future timesteps. Suitable for base feature generation
    such as random walks, AR(1) processes, and inter-feature influences where
    the target feature depends on its own or another feature's past values.
    """
    shape: Shape = field(default=Delta())
    timing: Timing = field(default=FixedDelay(1))

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> float:
        """
        Apply the rule at a single timestep t, reading only past values.

        Args:
            signals: dict of feature trajectories, partially filled up to t.
            t: current timestep being computed.

        Returns:
            float: contribution to target feature at timestep t.
        """
        triggers = []
        for inp in self.inputs:
            f = signals[inp.feature][:t]
            transformed = inp.transform(f)
            trigger = inp.threshold(transformed)
            triggers.append(trigger[-1] if len(trigger) > 0 else 0.0)

        combined = self.combination(triggers)
        timed = self.timing.apply_step(signals[self.target], t, combined)
        return float(self.magnitude(np.array([timed]))[0])

    def __post_init__(self):
        if not isinstance(self.timing, FixedDelay):
            raise ValueError(
                f"AutoRegressiveRule requires FixedDelay timing, got {type(self.timing).__name__}"
            )
        if self.timing.delay < 1:
            raise ValueError(
                "AutoRegressiveRule requires delay >= 1 to prevent cycles")
