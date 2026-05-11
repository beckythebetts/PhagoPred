import sympy as sp
from collections import defaultdict
from pathlib import Path

from PhagoPred.utils.logger import get_logger
from .rules import Var, Const, Add, Sub, Mul, Div, Rule, _vars
import numpy as np
import matplotlib.pyplot as plt

log = get_logger()


def get_variance_contribution(
        rule: Rule,
        noises_vals: dict[str:float]) -> dict[tuple[str, int], float]:
    vars_in_expr = _vars(rule.expr)
    context = {(v.name, v.lag): 0.0 for v in vars_in_expr}
    return {
        (v.name, v.lag):
        rule.expr.partial(v.name, v.lag, context)**2 * noises_vals[v.name]
        for v in vars_in_expr
    }


def estimate_variances(rules: list[Rule],
                       noise_vars: dict[str, float],
                       max_iter: int = 1000,
                       tol: float = 1e-6) -> dict[str, float]:
    rules_by_target = {rule.target: rule for rule in rules}
    var = dict(noise_vars)  # initialise to noise variance

    for _ in range(max_iter):
        new_var = dict(noise_vars)  # each feature always gets its own noise
        for feature, rule in rules_by_target.items():
            contribs = get_variance_contribution(rule, var)
            new_var[feature] = noise_vars.get(feature, 0.0) + sum(
                contribs.values())
        if max(abs(new_var[f] - var[f]) for f in var) < tol:
            break
        var = new_var

    return var


import numpy as np
import matplotlib.pyplot as plt


def _squared_derivs(rule: Rule) -> dict[tuple[str, int], float]:
    vars_in_expr = _vars(rule.expr)
    context = {(v.name, v.lag): 1e-6 for v in vars_in_expr}
    return {
        (v.name, v.lag): rule.expr.partial(v.name, v.lag, context)**2
        for v in vars_in_expr
    }


def compute_h_var(rules: list[Rule],
                  noise_vars: dict[str, float],
                  max_depth: int = 500,
                  tol: float = 1e-10) -> dict[str, np.ndarray]:
    """
    H_var[feature][k] = variance in feature from noise injected exactly k steps ago.
    Recurrence:
        H_var[feature][0] = noise_vars[feature]
        H_var[feature][k] = sum_{ source,lag } a^2 * H_var[source][k-lag]
    """
    sq_derivs = defaultdict(dict)
    for rule in rules:
        for (src, lag), a2 in _squared_derivs(rule).items():
            key = (src, lag)
            sq_derivs[rule.target][key] = sq_derivs[rule.target].get(key,
                                                                     0.0) + a2
    all_features = set(noise_vars) | {rule.target for rule in rules}
    H_var = {f: np.zeros(max_depth) for f in all_features}
    for f in all_features:
        H_var[f][0] = noise_vars.get(f, 0.0)

    for k in range(1, max_depth):
        for feature, derivs in sq_derivs.items():
            H_var[feature][k] = sum(a2 * H_var[src][k - lag]
                                    for (src, lag), a2 in derivs.items()
                                    if k >= lag and src in H_var)
        if all(H_var[f][k] < tol for f in all_features):
            break

    return H_var


def predictability_curve(rules: list[Rule],
                         target: str,
                         noise_vars: dict[str, float],
                         max_depth: int = 500,
                         max_horizon: int = 30) -> list[float]:
    """R²(H) for H = 1..max_horizon: fraction of Var(target) predictable from t-H."""
    H_var = compute_h_var(rules, noise_vars, max_depth)
    total = H_var[target].sum()
    if total == 0:
        return [0.0] * max_horizon
    return [
        float(H_var[target][H:].sum() / total)
        for H in range(1, max_horizon + 1)
    ]


def predictability_curve_features_only(rules: list[Rule],
                                       target: str,
                                       noise_vars: dict[str, float],
                                       max_depth: int = 500,
                                       max_horizon: int = 30) -> list[float]:
    H_var = compute_h_var(rules, noise_vars, max_depth)
    total = H_var[target].sum()

    rules_no_hazard_self = [
        r for r in rules if not (
            r.target == target and target in [v.name for v in _vars(r.expr)])
    ]

    H_var_observable = compute_h_var(
        rules_no_hazard_self,
        {
            **noise_vars, target: 0.0
        },  # no Hazard noise
        max_depth)[target]

    return [
        float(H_var_observable[H:].sum() / total)
        for H in range(1, max_horizon + 1)
    ]


def plot_predictability_curves(scenarios: dict[str, list[Rule]],
                               noise_vars: dict[str, float],
                               target: str = 'Hazard',
                               max_depth: int = 500,
                               max_horizon: int = 30,
                               save_path: str = None) -> None:
    horizons = list(range(1, max_horizon + 1))
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, rules in scenarios.items():
        r2 = predictability_curve_features_only(rules, target, noise_vars,
                                                max_depth, max_horizon)
        ax.plot(horizons, r2, marker='.', markersize=3, label=name)

    ax.set_xlabel('Forecast horizon (H)')
    ax.set_ylabel(f'R² — fraction of Var({target}) predictable from t−H')
    ax.set_title('Theoretical predictability ceiling by scenario')
    # ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        log.info(f'Saved to {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    from PhagoPred.survival_v2.data.graph_synthetic.scenarios import (
        _linear,
        _chain,
        _multiplicative,
        _ratio,
        _resetting_accumulation,
    )

    _noise_vars = {f: 1.0 for f in ['A', 'B', 'C', 'D', 'Hazard']}
    _noise_vars['Hazard'] = 1.0

    plot_predictability_curves(
        scenarios={
            'linear': _linear,
            'chain': _chain,
            'multiplicative': _multiplicative,
            'ratio': _ratio,
            'resetting_accumulation': _resetting_accumulation,
        },
        noise_vars=_noise_vars,
        target='Hazard',
        max_horizon=400,
        max_depth=500,
        save_path=Path('temp') / 'predictability_curves.png',
    )
