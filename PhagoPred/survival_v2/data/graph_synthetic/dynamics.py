from __future__ import annotations

import numpy as np
from scipy import sparse

from PhagoPred.survival_v2.utils.precision import ObservedPrecision
from .graph import CausalGraph
from .rules import Rule


def roll_forward(
    graph: CausalGraph,
    signals: dict[str, np.ndarray],
    start: int = 0,
    stop: int | None = None,
    rules: list[Rule] | None = None,
    frame_0: int = 0,
) -> None:
    """Step ``rules`` (default: all) over frames [start, stop), modifying
    ``signals`` IN PLACE."""
    if stop is None:
        stop = len(next(iter(signals.values())))
    for t in range(start, stop):
        graph.apply_rules(signals, t, frame_0, rules=rules)


def draw_noise(graph: CausalGraph,
               n_frames: int,
               n_samples: int,
               rng=None) -> dict[str, np.ndarray]:
    return {
        f.name:
        f.generate_signal(n_frames, n_samples,
                          rng).reshape(n_frames, n_samples)
        for f in graph.features
    }


def extend_signals(graph,
                   history,
                   new,
                   frame_0=0,
                   rules=None,
                   return_full=False):
    """Append ``new`` to ``history`` and step ``rules`` (default: all) over the
    new frames. ``frame_0`` is the index of observation frame 0 in ``history``.
    Returns history + new if ``return_full``, else only the new frames."""
    history_len = len(next(iter(history.values())))
    if history_len < graph.max_lag:
        raise ValueError(
            f'history has {history_len} frames, need {graph.max_lag}')
    full = {k: np.concatenate([history[k], new[k]], axis=0) for k in new}
    roll_forward(graph, full, history_len, rules=rules, frame_0=frame_0)
    return full if return_full else {
        k: v[history_len:]
        for k, v in full.items()
    }


def branch(
    graph: CausalGraph,
    trunk: dict[str, np.ndarray],
    branch_len: int,
    n_branches: int,
    frame_0: int = 0,
    rng=None,
) -> dict[str, np.ndarray]:
    history = {
        k: np.repeat(v[:, None], n_branches, axis=1)
        for k, v in trunk.items()
    }
    return extend_signals(graph,
                          history,
                          draw_noise(graph, branch_len, n_branches, rng),
                          frame_0=frame_0)


def linearised_precision(
    graph: CausalGraph,
    feature_names: list[str],
    base_signals: dict[str, np.ndarray],
    ridge: float = 1e-8,
) -> ObservedPrecision:
    """Gaussian precision over (feature, frame) nodes of ``feature_names`` frames, from the graph's own rules.
    """

    lengths = {len(base_signals[name]) for name in feature_names}
    if len(lengths) != 1:
        raise ValueError(f'base_signals have different lengths: {lengths}')
    n_frames = lengths.pop()

    col = {name: i * n_frames for i, name in enumerate(feature_names)}
    features = {f.name: f for f in graph.features}
    n_obs = len(feature_names) * n_frames

    rows, cols, vals, inv_sigma = [], [], [], []
    eq = 0
    for name in feature_names:
        pre_noise = features[name].pre_noise
        if pre_noise.kind != 'gaussian':
            raise ValueError(f'{name}: pre-noise is {pre_noise.kind!r}; the '
                             'precision needs Gaussian pre-noise')
        sigma = pre_noise.params['sigma']
        target_rules = [r for r in graph.rules if r.target == name]
        for rule in target_rules:
            missing = set(rule.inputs) - set(col)
            if missing:
                raise ValueError(
                    f'rule for {name} reads {missing}, which are not in '
                    f'feature_names; the conditional would be wrong')

        for t in range(n_frames):
            rows.append(eq)
            cols.append(col[name] + t)
            vals.append(1.0)
            for rule in target_rules:
                if t < rule.max_lag:
                    continue
                context = ({
                    (nm, lg): base_signals[nm][t - lg]
                    for nm, lgs in rule.inputs.items()
                    for lg in lgs
                })
                for nm, lgs in rule.inputs.items():
                    for lg in lgs:
                        coef = rule.expr.partial(nm, lg, context)
                        if abs(coef) > 1e-12:
                            rows.append(eq)
                            cols.append(col[nm] + t - lg)
                            vals.append(-float(coef))
            inv_sigma.append(1.0 / sigma)
            eq += 1

    incidence = sparse.csc_matrix((vals, (rows, cols)), shape=(eq, n_obs))
    D_sqrt = sparse.diags(np.asarray(inv_sigma))
    Q = (incidence.T @ D_sqrt @ D_sqrt @ incidence +
         ridge * sparse.eye(n_obs)).tocsc()
    return ObservedPrecision(n_obs, incidence, D_sqrt, Q, np.zeros(n_obs))
