from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from math import log

import numpy as np
import sympy as sp


class Expr(ABC):

    @abstractmethod
    def __call__(self, parent_values: dict[str, float]) -> float:
        ...

    @abstractmethod
    def partial(self, var_name: str, lag: int, context: dict) -> float:
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

    def partial(self, var_name, lag, context):
        return 1.0 if (self.name == var_name and self.lag == lag) else 0.0


class Const(Expr):

    def __init__(self, val):
        self.val = val

    def __call__(self, pv):
        return self.val

    def partial(self, var_name, lag, context):
        return 0.0


class BinaryOp(Expr, ABC):

    def __init__(self, left, right):
        self.left, self.right = left, right


class Add(BinaryOp):

    def __call__(self, context):
        return self.left(context) + self.right(context)

    def partial(self, var_name, lag, context):
        return (self.left.partial(var_name, lag, context) +
                self.right.partial(var_name, lag, context))


class Sub(BinaryOp):

    def __call__(self, context):
        return self.left(context) - self.right(context)

    def partial(self, var_name, lag, context):
        return (self.left.partial(var_name, lag, context) -
                self.right.partial(var_name, lag, context))


class Mul(BinaryOp):

    def __call__(self, context):
        return self.left(context) * self.right(context)

    def partial(self, var_name, lag, context):
        dl = self.left.partial(var_name, lag, context)
        dr = self.right.partial(var_name, lag, context)
        return dl * self.right(context) + self.left(context) * dr


class Div(BinaryOp):

    def __call__(self, context):
        return self.left(context) / self.right(context)

    def partial(self, var_name, lag, context):
        dl = self.left.partial(var_name, lag, context)
        dr = self.right.partial(var_name, lag, context)
        l, r = self.left(context), self.right(context)
        return (dl * r - l * dr) / r**2


class Pow(BinaryOp):

    def __call__(self, context):
        return self.left(context)**self.right(context)

    def partial(self, var_name, lag, context):
        dl = self.left.partial(var_name, lag, context)
        dr = self.right.partial(var_name, lag, context)
        l, r = self.left(context), self.right(context)
        result = r * l**(r - 1) * dl
        if l > 0 and dr != 0.0:
            result += l**r * log(l) * dr
        return result


class Min(Expr):

    def __init__(self, *sources: str | Expr):
        self.sources = [Var(s) if isinstance(s, str) else s for s in sources]

    def __call__(self, context):
        return min(s(context) for s in self.sources)

    def partial(self, var_name, lag, context):
        vals = [s(context) for s in self.sources]
        active = int(np.argmin(vals))
        return self.sources[active].partial(var_name, lag, context)


class Max(Expr):

    def __init__(self, *sources: str | Expr):
        self.sources = [Var(s) if isinstance(s, str) else s for s in sources]

    def __call__(self, context):
        return max(s(context) for s in self.sources)

    def partial(self, var_name, lag, context):
        vals = [s(context) for s in self.sources]
        active = int(np.argmax(vals))
        return self.sources[active].partial(var_name, lag, context)


class Apply(Expr):

    def __init__(self, func, source: str | Expr, *args, **kwargs):
        self.func = func
        self.source = Var(source) if isinstance(source, str) else source
        self.args = args
        self.kwargs = kwargs

    def __call__(self, context):
        return self.func(self.source(context), *self.args, **self.kwargs)

    def partial(self, var_name, lag, context):
        inner_deriv = self.source.partial(var_name, lag, context)
        x = self.source(context)
        eps = 1e-7
        func_deriv = (self.func(x + eps, *self.args, **self.kwargs) -
                      self.func(x - eps, *self.args, **self.kwargs)) / (2 * eps)
        return func_deriv * inner_deriv


class ReLU(Expr):

    def __init__(self, source: str | Expr, thresh: float = 0.0):
        self.source = Var(source) if isinstance(source, str) else source
        self.thresh = thresh

    def __call__(self, context):
        val = self.source(context)
        return val if val > self.thresh else 0.0

    def partial(self, var_name, lag, context):
        if self.source(context) > self.thresh:
            return self.source.partial(var_name, lag, context)
        return 0.0


class Threshold(Expr):

    def __init__(self, source: str | Expr, thresh: float = 0.0):
        self.source = Var(source) if isinstance(source, str) else source
        self.thresh = thresh

    def __call__(self, context):
        return 1.0 if self.source(context) > self.thresh else 0.0

    def partial(self, _var_name, _lag, _context):
        return 0.0


class Sigmoid(Expr):

    def __init__(self, source: str | Expr, k: float = 10.0, thresh: float = 0.0):
        self.source = Var(source) if isinstance(source, str) else source
        self.k = k
        self.thresh = thresh

    def __call__(self, context):
        val = self.source(context)
        return 1.0 / (1.0 + np.exp(-self.k * (val - self.thresh)))

    def partial(self, var_name, lag, context):
        s = self(context)
        return self.k * s * (1 - s) * self.source.partial(var_name, lag, context)


class Hill(Expr):

    def __init__(self, source: str | Expr, n: float = 2.0, K: float = 1.0):
        self.source = Var(source) if isinstance(source, str) else source
        self.n = n
        self.K = K

    def __call__(self, context):
        val = max(self.source(context), 0.0)
        return val**self.n / (self.K**self.n + val**self.n)

    def partial(self, var_name, lag, context):
        val = max(self.source(context), 0.0)
        if val == 0.0:
            return 0.0
        denom = self.K**self.n + val**self.n
        func_deriv = self.n * self.K**self.n * val**(self.n - 1) / denom**2
        return func_deriv * self.source.partial(var_name, lag, context)


class Abs(Expr):

    def __init__(self, source: str | Expr):
        self.source = Var(source) if isinstance(source, str) else source

    def __call__(self, context):
        return abs(self.source(context))

    def partial(self, var_name, lag, context):
        val = self.source(context)
        if val > 0:
            return self.source.partial(var_name, lag, context)
        elif val < 0:
            return -self.source.partial(var_name, lag, context)
        return 0.0


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
            return noise_term
        return noise_term + sum(
            expand_expr(rule.expr, time, depth - 1)
            for rule in rules_by_target[name])

    def expand_expr(expr, time, depth):
        if isinstance(expr, Const):
            return sp.Float(expr.val)
        elif isinstance(expr, Var):
            return expand_var(expr.name, time - expr.lag, depth)
        elif isinstance(expr, Add):
            return (expand_expr(expr.left, time, depth) +
                    expand_expr(expr.right, time, depth))
        elif isinstance(expr, Sub):
            return (expand_expr(expr.left, time, depth) -
                    expand_expr(expr.right, time, depth))
        elif isinstance(expr, Mul):
            return (expand_expr(expr.left, time, depth) *
                    expand_expr(expr.right, time, depth))
        elif isinstance(expr, Div):
            return (expand_expr(expr.left, time, depth) /
                    expand_expr(expr.right, time, depth))
        elif isinstance(expr, Apply):
            inner = expand_expr(expr.source, time, depth)
            return sp.Function(expr.func.__name__)(inner)
        elif hasattr(expr, 'source'):
            inner = expand_expr(expr.source, time, depth)
            return sp.Function(type(expr).__name__)(inner)
        return sp.Symbol(f'unknown_{time}')

    return sp.expand(expand_var(target, t, max_depth))