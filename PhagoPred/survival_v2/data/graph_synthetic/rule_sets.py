from __future__ import annotations
from dataclasses import dataclass, field

from .rules import (
    AutoCorrelationRule,
    Rule,
    Var,
    ReLU,
    Threshold,
)

FEATURES = ('A', 'B', 'C', 'D')

DEFAULT_AR_COEFF = 0.99
DEFAULT_K = 2.0
DEFAULT_LAGS = (50, 100, 150)
CHAIN_LAGS = (50, 50, 50)

FP_LAGS = (30, 40, 50)


def _gain(k: float, target_ar_coeff: float) -> float:
    return k * (1 - target_ar_coeff)


def _auto_correlate(ar_coeff: float = DEFAULT_AR_COEFF,
                    hazard_ar_coeff: float = DEFAULT_AR_COEFF,
                    features: tuple[str, ...] = FEATURES) -> list[Rule]:
    """AR(1) self-dependence: ar_coeff for each feature, hazard_ar_coeff for
    Hazard."""
    return ([AutoCorrelationRule(name, ar_coeff) for name in features] +
            [AutoCorrelationRule('Hazard', hazard_ar_coeff)])


def linear(ar_coeff: float = DEFAULT_AR_COEFF,
           hazard_ar_coeff: float = DEFAULT_AR_COEFF,
           lags: tuple = DEFAULT_LAGS,
           k: float = DEFAULT_K) -> list[Rule]:
    la, lb, lc = lags
    return [
        Rule(target='Hazard',
             expr=_gain(k, hazard_ar_coeff) *
             (Var('A', la) + Var('B', lb) + Var('C', lc)))
    ] + _auto_correlate(ar_coeff, hazard_ar_coeff)


def linear_chain(ar_coeff: float = DEFAULT_AR_COEFF,
                 hazard_ar_coeff: float = DEFAULT_AR_COEFF,
                 lags: tuple = CHAIN_LAGS,
                 k: float = DEFAULT_K) -> list[Rule]:
    lab, lbc, lch = lags
    return [
        Rule(target='B', expr=_gain(k, ar_coeff) * Var('A', lab)),
        Rule(target='C', expr=_gain(k, ar_coeff) * Var('B', lbc)),
        Rule(target='Hazard', expr=_gain(k, hazard_ar_coeff) * Var('C', lch)),
    ] + _auto_correlate(ar_coeff, hazard_ar_coeff)


def nonlinear_chain(ar_coeff: float = DEFAULT_AR_COEFF,
                    hazard_ar_coeff: float = DEFAULT_AR_COEFF,
                    lags: tuple = CHAIN_LAGS,
                    k: float = DEFAULT_K) -> list[Rule]:
    lab, lbc, lch = lags
    return [
        Rule(target='B', expr=_gain(k, ar_coeff) * ReLU(Var('A', lab))),
        Rule(target='C', expr=_gain(k, ar_coeff) * ReLU(Var('B', lbc))),
        Rule(target='Hazard',
             expr=_gain(k, hazard_ar_coeff) * ReLU(Var('C', lch))),
    ] + _auto_correlate(ar_coeff, hazard_ar_coeff)


def multiplicative_hazard(ar_coeff: float = DEFAULT_AR_COEFF,
                          hazard_ar_coeff: float = DEFAULT_AR_COEFF,
                          lags: tuple = DEFAULT_LAGS,
                          k: float = DEFAULT_K) -> list[Rule]:
    la, _, lb = lags
    return [
        Rule(target='Hazard',
             expr=_gain(k, hazard_ar_coeff) * Var('A', la) * Var('B', lb))
    ] + _auto_correlate(ar_coeff, hazard_ar_coeff)


def threshold_hazard(ar_coeff: float = DEFAULT_AR_COEFF,
                     hazard_ar_coeff: float = DEFAULT_AR_COEFF,
                     lags: tuple = DEFAULT_LAGS,
                     k: float = DEFAULT_K,
                     thresh: float = 6.0) -> list[Rule]:
    la, lb, lc = lags
    return [
        Rule(target='Hazard',
             expr=_gain(k, hazard_ar_coeff) * Threshold(
                 Var('A', la) + Var('B', lb) + Var('C', lc), thresh=thresh))
    ] + _auto_correlate(ar_coeff, hazard_ar_coeff)


RULE_SETS = {
    f.__name__: f
    for f in (linear, linear_chain, nonlinear_chain, multiplicative_hazard,
              threshold_hazard)
}
