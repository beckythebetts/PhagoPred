from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import sympy as sp


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


class AutoCorrelationRule(Rule):

    def __init__(self, feature: str, coeff: float, lag: int = 1):
        target = feature
        expr = coeff * Var(feature, lag)

        super().__init__(target, expr)


class MeanReversionRule(Rule):

    def __init__(self, feature: str, mean: float, theta: float):
        target = feature
        expr = theta * (mean - Var(feature))

        super().__init__(target, expr)


def expand_to_noise(rules: list[Rule],
                    target: str,
                    t: int,
                    max_depth: int = 10):
    rules_by_target = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.target].append(rule)

    def expand_var(name, time, depth):
        noise_term = sp.Symbol(f'noise_{name}_{time}')
        if depth <= 0 or name not in rules_by_target:
            return noise_term  # terminal — just the noise at this timestep
        # noise injected HERE plus the deterministic rule contributions expanded further back
        return noise_term + sum(
            expand_expr(rule.expr, time, depth - 1)
            for rule in rules_by_target[name])

    def expand_expr(expr, time, depth):
        if isinstance(expr, Const):
            return sp.Float(expr.val)
        elif isinstance(expr, Var):
            return expand_var(expr.name, time - expr.lag, depth)
        elif isinstance(expr, Add):
            return expand_expr(expr.left, time, depth) + expand_expr(
                expr.right, time, depth)
        elif isinstance(expr, Sub):
            return expand_expr(expr.left, time, depth) - expand_expr(
                expr.right, time, depth)
        elif isinstance(expr, Mul):
            return expand_expr(expr.left, time, depth) * expand_expr(
                expr.right, time, depth)
        elif isinstance(expr, Div):
            return expand_expr(expr.left, time, depth) / expand_expr(
                expr.right, time, depth)
        elif isinstance(expr, Apply):
            inner = expand_expr(expr.source, time, depth)
            return sp.Function(expr.func.__name__)(
                inner)  # opaque for nonlinear
        return sp.Symbol(f'unknown_{time}')

    return sp.expand(expand_var(target, t, max_depth))
