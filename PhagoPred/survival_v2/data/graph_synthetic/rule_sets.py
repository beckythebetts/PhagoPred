"""Rule sets for the graph-synthetic scenarios.

Features are AR(1) (``ar_coeff``) and run through the full burn-in. Hazard is
a decaying weighted window over its inputs (see ``hazard_window``): at frame t
it reads the inputs from frames t - lag - s, s = 0 .. memory - 1, and nothing
older. Combined with ``hazard_from_frame_0`` and landmarks >= the graph's
max_lag, the hazard over the prediction horizon then depends only on observed
frames, and maps features to risk the same way at every landmark.

Every lag must exceed the prediction horizon (25), so the last horizon frame
L + H reads frames before the landmark L (windows are [0, L)).
"""
from __future__ import annotations

from .rules import (
    AutoCorrelationRule,
    Expr,
    Rule,
    Var,
    ReLU,
    Threshold,
    shift_lags,
)

FEATURES = ('A', 'B', 'C', 'D')

DEFAULT_AR_COEFF = 0.99
DEFAULT_K = 2.0
DEFAULT_LAGS = (30, 55, 80)
CHAIN_LAGS = (50, 50, 50)
DEFAULT_MEMORY = 100

# Weight of the oldest frame in the hazard window, relative to the newest.
# Fixes the decay rate for a given memory, so the window shape is the same
# for every memory, just stretched.
TAIL_WEIGHT = 0.05

FP_LAGS = (30, 40, 50)


def _gain(k: float, target_ar_coeff: float) -> float:
    return k * (1 - target_ar_coeff)


def hazard_decay(memory: int) -> float:
    """psi such that psi ** (memory - 1) == TAIL_WEIGHT."""
    return TAIL_WEIGHT**(1.0 / (memory - 1))


def hazard_window(f: Expr, k: float = DEFAULT_K,
                  memory: int = DEFAULT_MEMORY) -> list[Rule]:
    """Rules making Hazard_t = g * sum_{s < memory} psi^s * f_{t - s}, where
    f_t is ``f`` evaluated with its lags as given (e.g. Var('A', 30) reads
    frame t - 30). g = k (1 - psi) / (1 - psi^memory) normalises the weights
    to sum to k.

    Written recursively, as three rules: the AR(1) term, the newest input,
    and the input leaving the window,

        H_t = psi H_{t-1} + g f_t - g psi^memory f_{t - memory}.
    """
    psi = hazard_decay(memory)
    g = k * (1 - psi) / (1 - psi**memory)
    return [
        Rule(target='Hazard', expr=g * f),
        Rule(target='Hazard', expr=(-g * psi**memory) * shift_lags(f, memory)),
        AutoCorrelationRule('Hazard', psi),
    ]


def _auto_correlate(ar_coeff: float = DEFAULT_AR_COEFF,
                    features: tuple[str, ...] = FEATURES) -> list[Rule]:
    """AR(1) self-dependence of each feature (Hazard's is in hazard_window)."""
    return [AutoCorrelationRule(name, ar_coeff) for name in features]


def linear(ar_coeff: float = DEFAULT_AR_COEFF,
           lags: tuple = DEFAULT_LAGS,
           k: float = DEFAULT_K,
           memory: int = DEFAULT_MEMORY) -> list[Rule]:
    la, lb, lc = lags
    return (hazard_window(Var('A', la) + Var('B', lb) + Var('C', lc), k,
                          memory) + _auto_correlate(ar_coeff))


def linear_chain(ar_coeff: float = DEFAULT_AR_COEFF,
                 lags: tuple = CHAIN_LAGS,
                 k: float = DEFAULT_K,
                 memory: int = DEFAULT_MEMORY) -> list[Rule]:
    lab, lbc, lch = lags
    return [
        Rule(target='B', expr=_gain(k, ar_coeff) * Var('A', lab)),
        Rule(target='C', expr=_gain(k, ar_coeff) * Var('B', lbc)),
    ] + hazard_window(Var('C', lch), k, memory) + _auto_correlate(ar_coeff)


def nonlinear_chain(ar_coeff: float = DEFAULT_AR_COEFF,
                    lags: tuple = CHAIN_LAGS,
                    k: float = DEFAULT_K,
                    memory: int = DEFAULT_MEMORY) -> list[Rule]:
    lab, lbc, lch = lags
    return [
        Rule(target='B', expr=_gain(k, ar_coeff) * ReLU(Var('A', lab))),
        Rule(target='C', expr=_gain(k, ar_coeff) * ReLU(Var('B', lbc))),
    ] + hazard_window(ReLU(Var('C', lch)), k, memory) + _auto_correlate(
        ar_coeff)


def multiplicative_hazard(ar_coeff: float = DEFAULT_AR_COEFF,
                          lags: tuple = DEFAULT_LAGS,
                          k: float = DEFAULT_K,
                          memory: int = DEFAULT_MEMORY) -> list[Rule]:
    la, _, lb = lags
    return (hazard_window(Var('A', la) * Var('B', lb), k, memory) +
            _auto_correlate(ar_coeff))


def threshold_hazard(ar_coeff: float = DEFAULT_AR_COEFF,
                     lags: tuple = DEFAULT_LAGS,
                     k: float = DEFAULT_K,
                     thresh: float = 6.0,
                     memory: int = DEFAULT_MEMORY) -> list[Rule]:
    la, lb, lc = lags
    return (hazard_window(
        Threshold(Var('A', la) + Var('B', lb) + Var('C', lc), thresh=thresh),
        k, memory) + _auto_correlate(ar_coeff))


RULE_SETS = {
    f.__name__: f
    for f in (linear, linear_chain, nonlinear_chain, multiplicative_hazard,
              threshold_hazard)
}
