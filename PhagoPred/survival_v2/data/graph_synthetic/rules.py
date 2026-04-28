from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class Expr(ABC):

    @abstractmethod
    def __call__(self, parent_values: dict[str, float]) -> float:
        ...

    def _wrap(self, other):
        return other if isinstance(other, Expr) else Const(other)

    def __add__(self, other):
        return Add(self, self._wrap(other))

    def __radd__(self, other):
        return Add(self._wrap(other), self)

    def __sub__(self, other):
        return Sub(self, self._wrap(other))

    def __rsub__(self, other):
        return Sub(self._wrap(other), self)

    def __mul__(self, other):
        return Mul(self, self._wrap(other))

    def __rmul__(self, other):
        return Mul(self._wrap(other), self)

    def __truediv__(self, other):
        return Div(self, self._wrap(other))

    def __rtruediv__(self, other):
        return Div(self._wrap(other), self)

    def __pow__(self, other):
        return Pow(self, self._wrap(other))

    def __rpow__(self, other):
        return Pow(self._wrap(other), self)


class Var(Expr):

    def __init__(self, name: str, lag: int = 1):
        self.name = name
        self.lag = lag

    def __call__(self, context):
        return context[(self.name, self.lag)]


class Const(Expr):

    def __init__(self, val):
        self.val = val

    def __call__(self, pv):
        return self.val


class BinaryOp(Expr, ABC):

    def __init__(self, left, right):
        self.left, self.right = left, right


class Add(BinaryOp):

    def __call__(self, context):
        return self.left(context) + self.right(context)


class Sub(BinaryOp):

    def __call__(self, context):
        return self.left(context) - self.right(context)


class Mul(BinaryOp):

    def __call__(self, context):
        return self.left(context) * self.right(context)


class Div(BinaryOp):

    def __call__(self, context):
        return self.left(context) / self.right(context)


class Pow(BinaryOp):

    def __call__(self, context):
        return self.left(context)**self.right(context)


class Min(Expr):

    def __init__(self, *sources: str | Expr):
        self.sources = [Var(s) if isinstance(s, str) else s for s in sources]

    def __call__(self, context):
        return min(s(context) for s in self.sources)


class Max(Expr):

    def __init__(self, *sources: str | Expr):
        self.sources = [Var(s) if isinstance(s, str) else s for s in sources]

    def __call__(self, context):
        return max(s(context) for s in self.sources)


class Apply(Expr):

    def __init__(self, func, source: str | Expr, *args, **kwargs):
        self.func = func
        self.source = Var(source) if isinstance(source, str) else source
        self.args = args
        self.kwargs = kwargs

    def __call__(self, context):
        return self.func(self.source(context), *self.args, **self.kwargs)


class Rule:

    def __init__(self, target: str, expr: Expr):
        self.target = target
        self.expr = expr

        self.max_lag = max(max(lags) for lags in self.get_inputs().values())

    def apply_step(self, signals: dict[str, np.ndarray], t: int) -> dict:
        if t < self.max_lag:
            return signals
        context = {
            (name, lag): signals[name][t - lag]
            for name, lags in self.get_inputs().items()
            for lag in lags
        }
        signals[self.target][t] += self.expr(context)
        return signals

    def get_inputs(self) -> dict[str, int]:
        """Get the lags of each variable in the rule expression"""
        result = {}
        for v in _vars(self.expr):
            if v.name not in result:
                result[v.name] = []
            result[v.name].append(v.lag)
        return result


def _vars(expr: Expr) -> list[Var]:
    if isinstance(expr, Var):
        return [expr]
    if isinstance(expr, BinaryOp):
        return _vars(expr.left) + _vars(expr.right)
    if hasattr(expr, 'source'):
        return _vars(expr.source)
    if hasattr(expr, 'sources'):
        return [v for s in expr.sources for v in _vars(s)]
    return []


def ReLU(val: float, thresh: float):
    if val > thresh:
        return val
    return 0


def threshold(val: float, thresh: float):
    if val > thresh:
        return 1
    return 0


def sigmoid(val: float, k: float = 10.0, thresh: float = 0.0) -> float:
    """Smooth switch; k controls steepness."""
    return 1.0 / (1.0 + np.exp(-k * (val - thresh)))


def hill(val: float, n: float = 2.0, K: float = 1.0) -> float:
    """Cooperative binding; n=Hill coefficient, K=EC50."""
    val = max(val, 0.0)
    return val**n / (K**n + val**n)


# class Apply(Expr):
#     def __init__(self, func, arg): self.func, self.arg = func, arg
#     def __call__(self, pv): return self.func(self.arg(pv))

# class StructuralForm(ABC):
#     """Base class.  compute() maps {feature: scalar} -> scalar increment."""

#     @abstractmethod
#     def compute(self, parent_values: dict[str, float]) -> float:
#         ...

#     def __call__(self, parent_values: dict[str, float]) -> float:
#         return self.compute(parent_values)

# class Sum(StructuralForm):
#     """g(\sum_i(f_i(x_i)))"""

#     def __init__(self, g: ComponentFunc, f_dict: dict[str, ComponentFunc]):
#         self.g = g
#         self.f_dict = f_dict

#     def compute(self, parent_values: dict[str, float]) -> float:
#         for name in self.f_dict:
#             if name not in parent_values:
#                 raise KeyError(
#                     f"Parent '{name}' missing from parent_values: {list(parent_values.keys())}"
#                 )

#         output = self.g(
#             sum([
#                 self.f_dict[name](parent_values[name]) for name in self.f_dict
#             ]))
#         return output

# class Product(StructuralForm):
#     """g(\product_i(f_i(x_i)))"""
#     def __init__(self, g)

# # ── Axis 1a: Single Index (monotone) ─────────────────────────────────────────

# class SingleIndex(StructuralForm):
#     """f(sum_i w_i * x_i)

#     All parents are first collapsed into one linear index, then a single
#     component function is applied.  The canonical single-cause or
#     dose–response model.

#     Args:
#         weights: {feature_name: weight}.  Missing features default to 1.0.
#         func:    ComponentFunc applied to the weighted sum.
#     """

#     def __init__(self, weights: dict[str, float], func: ComponentFunc):
#         self.weights = weights
#         self.func = func

#     def compute(self, parent_values: dict[str, float]) -> float:
#         index = sum(
#             self.weights.get(name, 1.0) * val
#             for name, val in parent_values.items())
#         return self.func(index)

# # ── Axis 1b: Additive Nonlinear ───────────────────────────────────────────────

# class AdditiveNonlinear(StructuralForm):
#     """sum_i f_i(x_i)

#     Each parent is transformed by its own component function, then the results
#     are summed.  Captures independent nonlinear contributions.

#     Args:
#         funcs: {feature_name: ComponentFunc}.
#                Features not in the dict fall through unchanged (identity).
#     """

#     def __init__(self, funcs: dict[str, ComponentFunc]):
#         self.funcs = funcs

#     def compute(self, parent_values: dict[str, float]) -> float:
#         total = 0.0
#         for name, val in parent_values.items():
#             f = self.funcs.get(name, Linear(slope=1.0, intercept=0.0))
#             total += f(val)
#         return total

# # ── Axis 1c: Interaction ──────────────────────────────────────────────────────

# class MultiplicativeInteraction(StructuralForm):
#     """prod_i f_i(x_i)
#     Args:
#         funcs: {feature_name: ComponentFunc}.
#     """

#     def __init__(self, funcs: dict[str, ComponentFunc]):
#         self.funcs = funcs

#     def compute(self, parent_values: dict[str, float]) -> float:
#         result = 1.0
#         for name, val in parent_values.items():
#             f = self.funcs.get(name, Linear(slope=1.0, intercept=0.0))
#             result *= f(val)
#         return result

# class ModulatedInteraction(StructuralForm):
#     """f_driver(x_driver) * g_modulator(x_modulator)

#     One parent drives the effect; another scales (gates) it.
#     Models conditional activation: the effect only appears when both the
#     driver signal and modulator are non-zero.

#     Args:
#         driver:    name of the driving parent feature.
#         modulator: name of the gating parent feature.
#         driver_func:    ComponentFunc for the driver.
#         modulator_func: ComponentFunc for the modulator.
#     """

#     def __init__(
#         self,
#         driver: str,
#         modulator: str,
#         driver_func: ComponentFunc,
#         modulator_func: ComponentFunc,
#     ):
#         self.driver = driver
#         self.modulator = modulator
#         self.driver_func = driver_func
#         self.modulator_func = modulator_func

#     def compute(self, parent_values: dict[str, float]) -> float:
#         d = self.driver_func(parent_values.get(self.driver, 0.0))
#         m = self.modulator_func(parent_values.get(self.modulator, 0.0))
#         return d * m

# class AdditiveWithInteraction(StructuralForm):
#     """sum_i f_i(x_i) + interaction_scale * prod_i x_i

#     Additive baseline plus a pairwise (or higher) interaction term.
#     The interaction_scale controls the strength of synergy/antagonism.

#     Args:
#         funcs:             {feature_name: ComponentFunc} for the additive part.
#         interaction_scale: weight on the product interaction term.
#     """

#     def __init__(
#         self,
#         funcs: dict[str, ComponentFunc],
#         interaction_scale: float = 1.0,
#     ):
#         self.funcs = funcs
#         self.interaction_scale = interaction_scale

#     def compute(self, parent_values: dict[str, float]) -> float:
#         additive = sum(
#             self.funcs.get(name, Linear(slope=1.0, intercept=0.0))(val)
#             for name, val in parent_values.items())
#         product = float(np.prod(list(parent_values.values())))
#         return additive + self.interaction_scale * product
