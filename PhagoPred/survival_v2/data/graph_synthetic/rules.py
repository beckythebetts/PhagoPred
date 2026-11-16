from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from math import log

import numpy as np

_EXPRS: dict[str, type[Expr]] = {}


class Expr(ABC):

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _EXPRS[cls.__name__] = cls

    def children(self) -> list[Expr]:
        out = []
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Expr):
                out.append(v)
            elif isinstance(v, list):
                out.extend(x for x in v if isinstance(x, Expr))
        return out

    def to_dict(self) -> dict:
        return {
            'op': type(self).__name__,
            **{
                f.name: _encode(getattr(self, f.name))
                for f in fields(self)
            }
        }

    @staticmethod
    def from_dict(d: dict) -> Expr:
        d = dict(d)
        cls = _EXPRS[d.pop('op')]
        return cls(**{k: _decode(v) for k, v in d.items()})

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


def _encode(v):
    if isinstance(v, Expr):
        return v.to_dict()
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _decode(v):
    if isinstance(v, dict) and 'op' in v:
        return Expr.from_dict(v)
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def _as_expr(source: str | Expr) -> Expr:
    """A feature name is shorthand for its lag-1 Var."""
    return Var(source) if isinstance(source, str) else source


@dataclass
class Var(Expr):
    name: str
    lag: int = 1

    def __call__(self, context):
        return context[(self.name, self.lag)]

    def partial(self, var_name, lag, context):
        return 1.0 if (self.name == var_name and self.lag == lag) else 0.0


@dataclass
class Const(Expr):
    val: float

    def __call__(self, pv):
        return self.val

    def partial(self, var_name, lag, context):
        return 0.0


@dataclass
class BinaryOp(Expr, ABC):
    left: Expr
    right: Expr


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


@dataclass
class Min(Expr):
    sources: list[Expr]

    def __post_init__(self):
        self.sources = [_as_expr(s) for s in self.sources]

    def __call__(self, context):
        return np.minimum.reduce([s(context) for s in self.sources])

    def partial(self, var_name, lag, context):
        vals = [s(context) for s in self.sources]
        active = int(np.argmin(vals))
        return self.sources[active].partial(var_name, lag, context)


@dataclass
class Max(Expr):
    sources: list[Expr]

    def __post_init__(self):
        self.sources = [_as_expr(s) for s in self.sources]

    def __call__(self, context):
        return np.maximum.reduce([s(context) for s in self.sources])

    def partial(self, var_name, lag, context):
        vals = [s(context) for s in self.sources]
        active = int(np.argmax(vals))
        return self.sources[active].partial(var_name, lag, context)


@dataclass
class Sin(Expr):
    source: Expr

    def __post_init__(self):
        self.source = _as_expr(self.source)

    def __call__(self, context):
        return np.sin(self.source(context))

    def partial(self, var_name, lag, context):
        # chain rule: d/dx sin(g) = cos(g) * dg/dx
        return np.cos(self.source(context)) * self.source.partial(
            var_name, lag, context)


@dataclass
class ReLU(Expr):
    source: Expr
    thresh: float = 0.0

    def __post_init__(self):
        self.source = _as_expr(self.source)

    def __call__(self, context):
        val = self.source(context)
        return np.where(val > self.thresh, val, 0.0)

    def partial(self, var_name, lag, context):
        if self.source(context) > self.thresh:
            return self.source.partial(var_name, lag, context)
        return 0.0


@dataclass
class Threshold(Expr):
    source: Expr
    thresh: float = 0.0

    def __post_init__(self):
        self.source = _as_expr(self.source)

    def __call__(self, context):
        return np.where(self.source(context) > self.thresh, 1.0, 0.0)

    def partial(self, _var_name, _lag, _context):
        return 0.0


@dataclass
class Sigmoid(Expr):
    source: Expr
    k: float = 10.0
    thresh: float = 0.0

    def __post_init__(self):
        self.source = _as_expr(self.source)

    def __call__(self, context):
        val = self.source(context)
        return 1.0 / (1.0 + np.exp(-self.k * (val - self.thresh)))

    def partial(self, var_name, lag, context):
        s = self(context)
        return self.k * s * (1 - s) * self.source.partial(
            var_name, lag, context)


@dataclass
class Hill(Expr):
    source: Expr
    n: float = 2.0
    K: float = 1.0

    def __post_init__(self):
        self.source = _as_expr(self.source)

    def __call__(self, context):
        val = np.maximum(self.source(context), 0.0)
        return val**self.n / (self.K**self.n + val**self.n)

    def partial(self, var_name, lag, context):
        val = max(self.source(context), 0.0)
        if val == 0.0:
            return 0.0
        denom = self.K**self.n + val**self.n
        func_deriv = self.n * self.K**self.n * val**(self.n - 1) / denom**2
        return func_deriv * self.source.partial(var_name, lag, context)


@dataclass
class Abs(Expr):
    source: Expr

    def __post_init__(self):
        self.source = _as_expr(self.source)

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

        self.inputs = self.get_inputs()
        self.max_lag = max(max(lags) for lags in self.inputs.values())

    def to_dict(self) -> dict:
        return {'target': self.target, 'expr': self.expr.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> Rule:
        return Rule(d['target'], Expr.from_dict(d['expr']))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self.target == other.target and self.expr == other.expr

    def __repr__(self) -> str:
        return f'Rule({self.target!r} += {self.expr!r})'

    def apply_step(self,
                   signals: dict[str, np.ndarray],
                   t: int,
                   start: int | None = None) -> dict:
        """Add this rule's contribution at step ``t``.

        If ``start`` is given the rule only acts from ``t >= start``, and
        inputs from before ``start`` contribute zero (the rule has no memory
        of anything before it starts)."""
        if t < self.max_lag:
            return signals
        if start is not None and t < start:
            return signals
        context = {}
        for name, lags in self.inputs.items():
            for lag in lags:
                value = signals[name][t - lag]
                if start is not None and t - lag < start:
                    value = 0.0 * value
                context[(name, lag)] = value
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
    return [v for child in expr.children() for v in _vars(child)]


class AutoCorrelationRule(Rule):

    def __init__(self, feature: str, coeff: float, lag: int = 1):
        target = feature
        expr = coeff * Var(feature, lag)

        super().__init__(target, expr)
