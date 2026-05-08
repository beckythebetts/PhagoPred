"""
Analytical predictability curves for synthetic graph scenarios.

Expands a target feature at time t into a SymPy expression purely in terms
of noise symbols (noise_<feature>_<timestep>), then partitions those terms
into predictable (noise injected at or before t-H) vs unpredictable (after),
and computes R² = Var(predictable) / Var(total) at each horizon H.
"""

import sympy as sp
from collections import defaultdict

from .rules import Var, Const, Add, Sub, Mul, Div, Apply, Rule


def expand_to_noise(rules: list[Rule], target: str = 'Hazard', t: int = 0,
                    max_depth: int = 20) -> sp.Expr:
    """
    Expand target feature at time t into a SymPy expression over noise symbols.

    Each noise injection at a node/timestep becomes an independent symbol
    noise_<feature>_<timestep>. The rules describe only the deterministic
    part of each update, so at every expansion step we also add the noise
    term for that node at that time.

    Args:
        rules: list of Rule objects defining the graph dynamics
        target: name of the feature to expand
        t: reference time (all noise timesteps are relative to this)
        max_depth: how many recursive steps to expand before terminating

    Returns:
        SymPy expression in noise symbols, expanded and simplified.
    """
    rules_by_target = defaultdict(list)
    for rule in rules:
        rules_by_target[rule.target].append(rule)

    def expand_var(name: str, time: int, depth: int, expanding: frozenset) -> sp.Expr:
        noise = sp.Symbol(f'noise_{name}_{time}')
        if depth <= 0 or name not in rules_by_target:
            return noise
        # stop self-loops: if we're already expanding this (name, time) stop here
        key = (name, time)
        if key in expanding:
            return noise
        return noise + sum(
            expand_expr(rule.expr, time, depth - 1, expanding | {key})
            for rule in rules_by_target[name]
        )

    def expand_expr(expr, time: int, depth: int, expanding: frozenset) -> sp.Expr:
        if isinstance(expr, Const):
            return sp.Float(expr.val)
        elif isinstance(expr, Var):
            return expand_var(expr.name, time - expr.lag, depth, expanding)
        elif isinstance(expr, Add):
            return (expand_expr(expr.left, time, depth, expanding)
                    + expand_expr(expr.right, time, depth, expanding))
        elif isinstance(expr, Sub):
            return (expand_expr(expr.left, time, depth, expanding)
                    - expand_expr(expr.right, time, depth, expanding))
        elif isinstance(expr, Mul):
            return (expand_expr(expr.left, time, depth, expanding)
                    * expand_expr(expr.right, time, depth, expanding))
        elif isinstance(expr, Div):
            # don't call sp.expand on divisions — keeps rational expressions tractable
            num = expand_expr(expr.left, time, depth, expanding)
            den = expand_expr(expr.right, time, depth, expanding)
            return sp.UnevaluatedExpr(num / den)
        elif isinstance(expr, Apply):
            inner = expand_expr(expr.source, time, depth, expanding)
            return sp.Function(expr.func.__name__)(inner)
        return sp.Symbol(f'unknown_{time}')

    return sp.expand(expand_var(target, t, max_depth, frozenset()))


def partition_at_horizon(expr: sp.Expr, t: int, H: int) -> tuple[sp.Expr, sp.Expr]:
    """
    Split expanded expression into predictable and unpredictable parts at horizon H.

    A term is predictable if ALL its noise symbols have timestep <= t-H,
    meaning they were injected before or at the prediction origin.
    Mixed terms (one factor before, one after cutoff) are unpredictable
    because E[noise_after | past] = 0.

    Args:
        expr: expanded SymPy expression from expand_to_noise
        t: reference time used in expand_to_noise
        H: forecast horizon

    Returns:
        (predictable, unpredictable) SymPy expressions
    """
    cutoff = t - H
    predictable = sp.Integer(0)
    unpredictable = sp.Integer(0)

    for term in sp.Add.make_args(expr):
        timesteps = [int(sym.name.split('_')[-1]) for sym in term.free_symbols]
        if all(ts <= cutoff for ts in timesteps):
            predictable += term
        else:
            unpredictable += term

    return predictable, unpredictable


def _variance(expr: sp.Expr, noise_sigma: float = 1.0) -> float:
    """
    Recursively compute variance of a SymPy expression in independent zero-mean noise.

    Exact for linear and product-of-noise expressions.
    Delta method approximation for opaque nonlinear functions (ReLU, threshold).

    Var(c * eps) = c^2 * sigma^2
    Var(eps_A * eps_B) = sigma^4           (independent zero-mean)
    Var(X + Y) = Var(X) + Var(Y)          (independent noise symbols, each appears once)
    Var(ReLU(X)) ≈ 0.5 * Var(X)          (delta method: P(X>0)=0.5 for zero-mean)
    Var(threshold(X)) ≈ 0.25              (Bernoulli(0.5) approximation)
    """
    if expr.is_number:
        return 0.0

    # single noise symbol
    if isinstance(expr, sp.Symbol):
        return noise_sigma ** 2 if str(expr).startswith('noise_') else 0.0

    # sum: independent terms after expand() so variances add
    if isinstance(expr, sp.Add):
        return sum(_variance(arg, noise_sigma) for arg in expr.args)

    # product: separate constant coefficient, bare noise symbols, complex factors
    if isinstance(expr, sp.Mul):
        const = 1.0
        noise_syms = []
        complex_factors = []
        for arg in expr.args:
            if arg.is_number:
                const *= float(arg)
            elif isinstance(arg, sp.Symbol) and str(arg).startswith('noise_'):
                noise_syms.append(arg)
            else:
                complex_factors.append(arg)

        # c * eps_1 * eps_2 * ... — exact for zero-mean independent noise
        if not complex_factors:
            return const ** 2 * noise_sigma ** (2 * len(noise_syms))

        # single stochastic factor — Var(c * X) = c^2 * Var(X)
        all_stochastic = noise_syms + complex_factors
        if len(all_stochastic) == 1:
            f = all_stochastic[0]
            base = noise_sigma ** 2 if isinstance(f, sp.Symbol) else _variance(f, noise_sigma)
            return const ** 2 * base

        # multiple stochastic factors — Var(X*Y) ≈ Var(X)*Var(Y) for zero-mean
        var_product = 1.0
        for f in all_stochastic:
            var_product *= noise_sigma ** 2 if isinstance(f, sp.Symbol) else _variance(f, noise_sigma)
        return const ** 2 * var_product

    # opaque function — delta method on the inner expression
    if expr.args:
        inner = expr.args[0]
        inner_var = _variance(inner, noise_sigma)
        func_name = type(expr).__name__
        if func_name == 'ReLU':
            return 0.5 * inner_var   # P(X > 0) ≈ 0.5 for zero-mean input
        if func_name == 'threshold':
            return 0.25              # Var(Bernoulli(0.5))
        return inner_var             # identity approximation for unknown functions

    return 0.0


def r2_at_horizon(expr: sp.Expr, t: int, H: int, noise_sigma: float = 1.0) -> float:
    """
    R² at forecast horizon H: fraction of variance in target explained by
    all features up to t-H.

    R² = Var(predictable) / Var(total)

    Args:
        expr: expanded SymPy expression from expand_to_noise
        t: reference time
        H: forecast horizon
        noise_sigma: standard deviation of noise (cancels in ratio if uniform)

    Returns:
        R² in [0, 1]
    """
    pred, _ = partition_at_horizon(expr, t, H)

    var_pred = _variance(pred, noise_sigma)
    var_total = _variance(expr, noise_sigma)

    return var_pred / var_total if var_total > 0 else 0.0


def predictability_curve(rules: list[Rule], target: str = 'Hazard',
                         max_depth: int = 20, max_horizon: int = 30,
                         noise_sigma: float = 1.0) -> list[float]:
    """
    Compute the theoretical R² predictability curve over forecast horizons.

    Args:
        rules: list of Rule objects defining the graph dynamics
        target: feature whose predictability to analyse
        max_depth: expansion depth (should be >= max_horizon)
        max_horizon: maximum forecast horizon to compute
        noise_sigma: noise standard deviation (only matters for mixed-order terms)

    Returns:
        List of R² values for H = 1, 2, ..., max_horizon
    """
    expr = expand_to_noise(rules, target=target, t=0, max_depth=max_depth)
    return [
        r2_at_horizon(expr, t=0, H=h, noise_sigma=noise_sigma)
        for h in range(1, max_horizon + 1)
    ]


def plot_predictability_curves(
    scenarios: dict[str, list[Rule]],
    target: str = 'Hazard',
    max_depth: int = 20,
    max_horizon: int = 30,
    noise_sigma: float = 1.0,
    save_path: str = None,
) -> None:
    """
    Plot theoretical R² predictability curves for multiple scenarios.

    Args:
        scenarios: dict mapping scenario name to list of Rules
        target: feature to analyse
        max_depth: expansion depth passed to predictability_curve
        max_horizon: maximum horizon to plot
        noise_sigma: noise standard deviation
        save_path: if given, save figure to this path; otherwise show interactively
    """
    import matplotlib.pyplot as plt

    horizons = list(range(1, max_horizon + 1))
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, rules in scenarios.items():
        print(f"Computing {name}...")
        r2 = predictability_curve(rules, target=target, max_depth=max_depth,
                                  max_horizon=max_horizon, noise_sigma=noise_sigma)
        ax.plot(horizons, r2, marker='o', markersize=3, label=name)

    ax.set_xlabel('Forecast horizon (H)')
    ax.set_ylabel(f'R²  —  fraction of Var({target}) predictable from t−H')
    ax.set_title('Theoretical predictability ceiling by scenario')
    ax.set_ylim(0, 1)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    from PhagoPred.survival_v2.data.graph_synthetic.scenarios import (
        _linear, _chain, _multiplicative, _ratio, _resetting_accumulation,
    )

    plot_predictability_curves(
        scenarios={
            'linear':                 _linear,
            'chain':                  _chain,
            'multiplicative':         _multiplicative,
            'ratio':                  _ratio,
            'resetting_accumulation': _resetting_accumulation,
        },
        max_horizon=20,
        max_depth=25,
        save_path='predictability_curves.png',
    )
