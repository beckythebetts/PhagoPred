"""
Axis 1 — Structural Form
========================
Describes how a set of parent values combine into a single scalar increment.

Each class receives a dict {feature_name: value} and returns a float.

Three families:
  SingleIndex        — f(w1*x1 + w2*x2 + ...)       linear combination first, then one function
  AdditiveNonlinear  — f1(x1) + f2(x2) + ...         each parent gets its own function
  Interaction        — cross-terms / products          non-additive coupling
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from .component_funcs import ComponentFunc, Linear


class StructuralForm(ABC):
    """Base class.  compute() maps {feature: scalar} -> scalar increment."""

    @abstractmethod
    def compute(self, parent_values: dict[str, float]) -> float: ...

    def __call__(self, parent_values: dict[str, float]) -> float:
        return self.compute(parent_values)


# ── Axis 1a: Single Index (monotone) ─────────────────────────────────────────

class SingleIndex(StructuralForm):
    """f(sum_i w_i * x_i)

    All parents are first collapsed into one linear index, then a single
    component function is applied.  The canonical single-cause or
    dose–response model.

    Args:
        weights: {feature_name: weight}.  Missing features default to 1.0.
        func:    ComponentFunc applied to the weighted sum.
    """

    def __init__(self, weights: dict[str, float], func: ComponentFunc):
        self.weights = weights
        self.func = func

    def compute(self, parent_values: dict[str, float]) -> float:
        index = sum(
            self.weights.get(name, 1.0) * val
            for name, val in parent_values.items()
        )
        return self.func(index)


# ── Axis 1b: Additive Nonlinear ───────────────────────────────────────────────

class AdditiveNonlinear(StructuralForm):
    """sum_i f_i(x_i)

    Each parent is transformed by its own component function, then the results
    are summed.  Captures independent nonlinear contributions.

    Args:
        funcs: {feature_name: ComponentFunc}.
               Features not in the dict fall through unchanged (identity).
    """

    def __init__(self, funcs: dict[str, ComponentFunc]):
        self.funcs = funcs

    def compute(self, parent_values: dict[str, float]) -> float:
        total = 0.0
        for name, val in parent_values.items():
            f = self.funcs.get(name, Linear(slope=1.0, intercept=0.0))
            total += f(val)
        return total


# ── Axis 1c: Interaction ──────────────────────────────────────────────────────

class MultiplicativeInteraction(StructuralForm):
    """prod_i f_i(x_i)

    Product of per-parent component functions.  Output is zero if any
    factor is zero — a quantitative AND gate.

    Args:
        funcs: {feature_name: ComponentFunc}.
    """

    def __init__(self, funcs: dict[str, ComponentFunc]):
        self.funcs = funcs

    def compute(self, parent_values: dict[str, float]) -> float:
        result = 1.0
        for name, val in parent_values.items():
            f = self.funcs.get(name, Linear(slope=1.0, intercept=0.0))
            result *= f(val)
        return result


class ModulatedInteraction(StructuralForm):
    """f_driver(x_driver) * g_modulator(x_modulator)

    One parent drives the effect; another scales (gates) it.
    Models conditional activation: the effect only appears when both the
    driver signal and modulator are non-zero.

    Args:
        driver:    name of the driving parent feature.
        modulator: name of the gating parent feature.
        driver_func:    ComponentFunc for the driver.
        modulator_func: ComponentFunc for the modulator.
    """

    def __init__(
        self,
        driver: str,
        modulator: str,
        driver_func: ComponentFunc,
        modulator_func: ComponentFunc,
    ):
        self.driver = driver
        self.modulator = modulator
        self.driver_func = driver_func
        self.modulator_func = modulator_func

    def compute(self, parent_values: dict[str, float]) -> float:
        d = self.driver_func(parent_values.get(self.driver, 0.0))
        m = self.modulator_func(parent_values.get(self.modulator, 0.0))
        return d * m


class AdditiveWithInteraction(StructuralForm):
    """sum_i f_i(x_i) + interaction_scale * prod_i x_i

    Additive baseline plus a pairwise (or higher) interaction term.
    The interaction_scale controls the strength of synergy/antagonism.

    Args:
        funcs:             {feature_name: ComponentFunc} for the additive part.
        interaction_scale: weight on the product interaction term.
    """

    def __init__(
        self,
        funcs: dict[str, ComponentFunc],
        interaction_scale: float = 1.0,
    ):
        self.funcs = funcs
        self.interaction_scale = interaction_scale

    def compute(self, parent_values: dict[str, float]) -> float:
        additive = sum(
            self.funcs.get(name, Linear(slope=1.0, intercept=0.0))(val)
            for name, val in parent_values.items()
        )
        product = float(np.prod(list(parent_values.values())))
        return additive + self.interaction_scale * product
