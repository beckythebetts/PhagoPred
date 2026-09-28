from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, fields, field
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
from scipy import sparse as _sparse

from PhagoPred.survival_v2.data.graph_synthetic import CausalGraph
from PhagoPred.survival_v2.interpret import (SampleWithSHAP, SHAPResult,
                                             ExplainerEnum, BackgroundEnum)
from PhagoPred.survival_v2.interpret.precision import (ObservedPrecision,
                                                       condition_and_sample)


@dataclass
class outputs:
    hazard: np.ndarray
    pmf: np.ndarray
    cdf: np.ndarray

    def expected_time(self, hazard_bins: np.ndarray | None = None):
        # === CHANGE TO REDCUCED MEAN SURVIVAL TIME ===
        survival = 1.0 - np.cumsum(self.pmf, axis=0)
        if hazard_bins is not None:
            bin_widths = np.diff(hazard_bins)
            t = (survival * bin_widths[:, None]).sum(axis=0)
        else:
            t = survival.sum(axis=0)
        return t

    def binary_prob(self):
        return self.cdf[-1]

    def get_target(
        self,
        output_type: Literal['expected_time', 'binary'],
        hazard_bins: np.ndarray | None = None,
    ) -> float | np.ndarray:
        if output_type == 'expected_time':
            return self.expected_time(hazard_bins)
        else:
            return self.binary_prob()


@dataclass
class baseSample:
    # (num_feats, ctx + seq_len), row order == graph.features: ``ctx`` frames
    # of burn-in context, then the observation window (frame 0 at ``ctx``).
    signals: np.ndarray
    noise: np.ndarray  # (num_feats, ctx + seq_len)
    lf: int  # landmark, relative to the window
    death_frame: int
    ctx: int = 0


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    sf = np.cumprod(1.0 - hazards, axis=0)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def _pre_scale(noise) -> float:
    """Std/scale of a feature's pre-noise, for the Gaussian bridge variance.

    Only the proposal uses this Gaussian; the weight scores the true pre-noise
    density, so a mismatched scale (or a non-Gaussian pre-noise) costs ESS, never
    correctness.
    """
    return float(getattr(noise, 'sigma', getattr(noise, 'scale', 1.0)))


def _apply_hazard_rules(graph: CausalGraph,
                        signals: dict[str, np.ndarray],
                        time_steps: int,
                        start: int = 0) -> dict:
    hazard_rules = [r for r in graph.rules if r.target == 'Hazard']
    for t in range(start, time_steps):
        for rule in hazard_rules:
            rule.apply_step(signals, t)
    return (signals)


def _outputs_from_hazard(hazard: np.ndarray,
                         hazard_bins: np.ndarray | None = None) -> outputs:
    cdf = np.cumsum(_pmf_from_hazards(hazard), axis=0)  # (horizon, B)

    if hazard_bins is not None:
        bins = np.asarray(hazard_bins, dtype=int)
        hazard = np.stack([
            1.0 - np.prod(1.0 - hazard[bins[i]:bins[i + 1]], axis=0)
            for i in range(len(bins) - 1)
        ])  # (n_bins, B)
    else:
        hazard = hazard  # (horizon, B)

    pmf = _pmf_from_hazards(hazard)
    return outputs(hazard, pmf, cdf)


def _landmark_sample(
    signals: dict,
    hazard_calibration_func: callable,
    max_sequence_length: int,
    min_sequence_length: int,
):
    base_hazard = hazard_calibration_func(signals['Hazard'])

    # Sample death frame
    base_pmf = _pmf_from_hazards(base_hazard)
    base_cif = np.cumsum(base_pmf)
    u = np.random.rand()
    if u <= base_cif.max():
        death_frame = float(np.argmax(base_cif >= u))
    else:
        death_frame = None

    # Sample landmark frames
    lf = np.arange(max_sequence_length)
    lf = lf[lf > min_sequence_length]
    if death_frame is not None:
        lf = lf[lf <= death_frame]
    if len(lf) == 0:
        return None
    lf = np.random.choice(lf)

    return lf, death_frame


def _generate_base_sample(graph: CausalGraph,
                          seq_len: int,
                          horizon: int,
                          min_seq_len: int,
                          hazard_calibration_func: callable,
                          ctx: int = 0) -> baseSample | None:
    """One trajectory from the same process as the datasets (burn-in and
    survivor conditioning via ``graph.simulate``), keeping the last ``ctx``
    burn-in frames as context before the window."""
    sig, noise = graph.simulate(1,
                                seq_len + horizon,
                                hazard_func=hazard_calibration_func,
                                keep_context=ctx,
                                post_noise=False,
                                return_noise=True)
    window = {name: arr[ctx:, 0] for name, arr in sig.items()}
    landmark_sample = _landmark_sample(
        window,
        hazard_calibration_func,
        seq_len,
        min_seq_len,
    )
    if landmark_sample is None:
        return None

    lf, death_frame = landmark_sample

    # Dict -> array only here, once: everything downstream of ``baseSample``
    # (and ``SampleWithSHAP``) works in array form.
    signals_arr = np.stack([sig[f.name][:, 0] for f in graph.features])
    noise_arr = np.stack([noise[f.name][:, 0] for f in graph.features])

    return baseSample(signals_arr, noise_arr, lf, death_frame, ctx)


def _realised_horizon_cif(hazard_signal: np.ndarray, lf: int, horizon: int,
                          hazard_calibration_func: callable) -> float:
    """CIF over [lf, lf+horizon] along the realised base trajectory.

    Cheap proxy for whether a sample's target (RMST / binary CIF) sits in a
    non-degenerate regime. A censored cell far from death has CIF≈0, so every
    perturbation leaves the target saturated (RMST at its max) and the Shapley
    values collapse to ≈0 — uninformative for the comparison. Filtering on this
    avoids drawing such degenerate samples.
    """
    hz = np.clip(hazard_calibration_func(hazard_signal[lf:lf + horizon]), 0.0,
                 1.0)
    return float(1.0 - np.prod(1.0 - hz))


def _contributions_from_values(values: np.ndarray,
                               permutation_order: np.ndarray) -> np.ndarray:
    """Marginals of ``values`` (n_players + 1,), the value of the first k
    players of ``permutation_order`` for k = 0..n_players, scattered back to
    player order."""
    contributions = np.zeros(permutation_order.shape[0])
    contributions[permutation_order] = np.diff(values)
    return contributions


# ─────────────────── joint-conditional (RTS/FFBS) observational ───────────────
# The per-frame ``_observational`` bridge is exact only for a *decoupled* observed
# subgraph (each feature its own AR chain — base_linear/multiplicative/ratio). When
# observed features drive one another (base_chain: A->B->C) it degenerates. The
# joint estimator below samples the free past-nodes from the exact Gaussian
# conditional over the whole observed subgraph at once, so coupling is handled.
#
# Tier-1/2 fast path: sparse LU per coalition (not dense Cholesky), sampling via
# the structural square-root (draw w ~ N(0, Q_FF) from the incidence, then
# Q_FF^{-1} w); no reweighting when the observed subgraph is linear (weights are
# then constant); common random numbers across coalitions so a dummy feature's
# marginal is *identically* zero.


@dataclass
class _GroundTruthPrecision:
    """A shared ``ObservedPrecision`` (see ``interpret.precision``) plus the
    ground-truth-specific metadata: which nodes are observed vs. Hazard, and
    whether the linearisation used to build ``prec.Q`` is exact (linear rules)
    or only a local proposal needing importance-reweighting (nonlinear rules)
    — a distinction that doesn't apply to a VAR-estimated precision, which is
    itself the working model rather than a proposal for some known truth."""
    obs: list  # observed (non-Hazard) feature names, in graph order
    lf: int
    is_linear: bool
    prec: ObservedPrecision


def _observed_is_linear(graph: CausalGraph) -> bool:
    """True iff no rule targeting an *observed* feature is nonlinear.
    """
    from PhagoPred.survival_v2.data.graph_synthetic import rules as R
    nonlin = (R.ReLU, R.Sigmoid, R.Hill, R.Threshold, R.Apply, R.Min, R.Max,
              R.Abs, R.Pow)

    def walk(e):
        yield e
        if isinstance(e, R.BinaryOp):
            yield from walk(e.left)
            yield from walk(e.right)
        if hasattr(e, 'source'):
            yield from walk(e.source)
        if hasattr(e, 'sources'):
            for s in e.sources:
                yield from walk(s)

    for rule in graph.rules:
        if rule.target == 'Hazard':
            continue
        for node in walk(rule.expr):
            if isinstance(node, nonlin):
                return False
            if isinstance(node,
                          (R.Mul, R.Div)):  # product of two Vars = nonlinear
                lv = any(isinstance(x, R.Var) for x in walk(node.left))
                rv = any(isinstance(x, R.Var) for x in walk(node.right))
                if lv and rv:
                    return False
    return True


def _build_observed_precision(graph: CausalGraph,
                              base_signals: np.ndarray,
                              lf: int,
                              ridge: float = 1e-8) -> _GroundTruthPrecision:
    """Linear-Gaussian precision over observed past nodes, once per base sample.

    Each observed node (f, t) contributes an innovation equation
    ``eps = x[f,t] - sum_parents coef * x[parent, t-lag]``.  For a nonlinear rule
    ``coef`` is the EKF Jacobian (``Expr.partial``) at the *base* trajectory, so
    the precision is the linearised proposal; the exact rule is used later in the
    importance weight.  Node column == full-player index because the observed
    (non-Hazard) features come first in ``graph.features``.
    """
    obs = [f.name for f in graph.features if f.name != 'Hazard']
    feat_idx = {f.name: i for i, f in enumerate(graph.features)}
    col = {(f, t): fi * lf + t for fi, f in enumerate(obs) for t in range(lf)}
    n_obs = len(col)
    scales = {f.name: _pre_scale(f.pre_noise) for f in graph.features}
    rows, cols, vals, sqrt_iv = [], [], [], []
    eq = 0
    for f in obs:
        for t in range(lf):
            rows.append(eq)
            cols.append(col[(f, t)])
            vals.append(1.0)
            for rule in graph.rules:
                if rule.target != f or t < rule.max_lag:
                    continue
                ctx = {
                    (nm, lg): base_signals[feat_idx[nm], t - lg]
                    for nm, lgs in rule.inputs.items()
                    for lg in lgs
                }
                for nm, lgs in rule.inputs.items():
                    for lg in lgs:
                        if t - lg < 0:
                            continue
                        c = rule.expr.partial(nm, lg, ctx)
                        if abs(c) > 1e-12:
                            rows.append(eq)
                            cols.append(col[(nm, t - lg)])
                            vals.append(-float(c))
            sqrt_iv.append(1.0 / scales[f])
            eq += 1
    incidence = _sparse.csc_matrix((vals, (rows, cols)), shape=(eq, n_obs))
    D_sqrt = _sparse.diags(
        np.asarray(sqrt_iv))  # diagonal: independent per-feature noise
    Q = (incidence.T @ D_sqrt @ D_sqrt @ incidence +
         ridge * _sparse.eye(n_obs)).tocsc()
    prec = ObservedPrecision(n_obs, incidence, D_sqrt, Q, np.zeros(n_obs))
    return _GroundTruthPrecision(obs, lf, _observed_is_linear(graph), prec)


def _hazard_history(base_signals_row: np.ndarray, base_noise_row: np.ndarray,
                    seq_len: int, n_boundary: int, B: int) -> np.ndarray:
    """Hazard array ready for its rules to be re-applied from ``n_boundary``:
    the boundary frames hold the base Hazard (the recursion's starting state)
    and every later frame its base noise."""
    hazard = np.repeat(base_noise_row[:seq_len, None], B, axis=1)
    hazard[:n_boundary] = base_signals_row[:n_boundary, None]
    return hazard


def _propagate_and_target(graph,
                          past,
                          base_signals,
                          base_noise,
                          n_past,
                          n_boundary,
                          seq_len,
                          hazard_calibration_func,
                          hazard_bins,
                          output_type,
                          B,
                          survival_start: int | None = None):
    """Fill observed past with ``past`` samples, propagate the horizon, read target.

    Returns ``(target, log_survival)``: ``log_survival`` (B,) is each draw's
    log-probability of surviving frames [survival_start, n_past), i.e. of
    being alive at the landmark as the sample is known to be; zeros if
    ``survival_start`` is None.

    All arrays are in extended coordinates: burn-in context, then the window.
    ``past`` (feature-name-keyed, required by ``rule.apply_step``) covers
    frames [0, n_past) = context + window up to the landmark; ``base_signals``
    and ``base_noise`` are the sample-level arrays.

    Hazard is not a player, so it has no entry in ``past``; its history is
    rebuilt from the boundary state through the sampled observed past (as in
    ``_interventional_value``). Leaving it zero would drop its autoregressive
    carry-over into the horizon, so v(all pinned) would not reproduce the base
    sample's own target.
    """
    sig = {}
    for f_idx, f in enumerate(graph.features):
        if f.name in past:
            s = np.empty((seq_len, B))
            s[:n_past] = past[f.name]
            s[n_past:] = base_noise[f_idx, n_past:seq_len, None]
        else:
            s = _hazard_history(base_signals[f_idx], base_noise[f_idx],
                                seq_len, n_boundary, B)
        sig[f.name] = s
    _apply_hazard_rules(graph, sig, n_past, start=n_boundary)
    log_survival = np.zeros(B)
    if survival_start is not None and survival_start < n_past:
        past_rates = np.clip(
            hazard_calibration_func(sig['Hazard'][survival_start:n_past]),
            0.0, 1.0 - 1e-12)
        log_survival = np.log1p(-past_rates).sum(axis=0)
    for t in range(n_past, seq_len):
        for rule in graph.rules:
            rule.apply_step(sig, t)
    hazard = hazard_calibration_func(sig['Hazard'][n_past:seq_len])
    target = _outputs_from_hazard(hazard, hazard_bins).get_target(
        output_type, hazard_bins)  # (B,)
    return target, log_survival


def _true_log_prior(graph, past, prec: _GroundTruthPrecision,
                    B: int) -> np.ndarray:
    """Sum of *true* (nonlinear) innovation log-densities over observed past nodes."""
    lf = prec.lf
    scales = {f.name: _pre_scale(f.pre_noise) for f in graph.features}
    logp = np.zeros(B)
    for f in prec.obs:
        var = scales[f]**2
        norm = -0.5 * np.log(2 * np.pi * var)
        for t in range(lf):
            drift = np.zeros(B)
            for rule in graph.rules:
                if rule.target != f or t < rule.max_lag:
                    continue
                ctx = {
                    (nm, lg): past[nm][t - lg]
                    for nm, lgs in rule.inputs.items()
                    for lg in lgs
                }
                drift = drift + rule.expr(ctx)
            eps = past[f][t] - drift
            logp += -0.5 * eps * eps / var + norm
    return logp


def _coalition_value(prec: _GroundTruthPrecision, pinned_mask: np.ndarray,
                     basevec: np.ndarray, xi: np.ndarray, graph, base_signals,
                     base_noise, seq_len, n_boundary, hazard_calibration_func,
                     hazard_bins, output_type, B,
                     survival_start: int | None = None) -> float:
    """v(S) = E[target | observed pinned = base, alive at landmark] via
    joint-conditional samples.

    ``prec.lf`` here is the number of past frames in extended coordinates
    (context + window up to the landmark); ``pinned_mask`` covers all of them,
    with the context already set (boundary pinned, rest free). The Gaussian
    conditional ignores that the cell survived to the landmark, so draws are
    importance-weighted by their survival probability from
    ``survival_start`` (None: no survival conditioning)."""
    lf, obs = prec.lf, prec.obs
    result = condition_and_sample(prec.prec, pinned_mask, basevec, xi)
    # result.values is (n_obs, B), feature-major (col[(f,t)] = fi*lf+t) and
    # already has pinned rows = basevec, free rows = conditional samples — no
    # separate initialisation from base_signals, and no _col_to_ft scatter,
    # needed: each feature's full (lf, B) block is just a contiguous slice.
    past = {
        f: result.values[fi * lf:(fi + 1) * lf]
        for fi, f in enumerate(obs)
    }
    logw = np.zeros(B)
    if len(
            result.F
    ) > 0 and not prec.is_linear:  # reweight the EKF-linearised proposal
        mean = result.lu.solve(result.rhs)
        Z = result.values[result.F]  # mean_vec is zero for ground truth, so
        #                              this equals the raw solve output ``Z``.
        d = Z - mean[:, None]
        logdet = float(np.sum(np.log(np.abs(result.lu.U.diagonal()))))
        logq = (0.5 * logdet - 0.5 * np.einsum('ib,ib->b', d, result.Qff @ d) -
                0.5 * len(result.F) * np.log(2 * np.pi))
        logw = _true_log_prior(graph, past, prec, B) - logq
    tgt, log_survival = _propagate_and_target(graph, past, base_signals,
                                              base_noise, lf, n_boundary,
                                              seq_len, hazard_calibration_func,
                                              hazard_bins, output_type, B,
                                              survival_start)
    logw = logw + log_survival
    ww = np.exp(logw - logw.max())
    return float((ww * tgt).sum() / ww.sum())


# ─────────────────── the three attribution axes / games ──────────────────────
# KernelSHAP computes three *separate* Shapley games at different mask
# granularities; the ground truth mirrors them so each axis is directly
# comparable. A "player" masks (pins/frees) a set of (feature, frame) nodes:
#   'temporal_feature' : one (feature, segment) per player  -> the joint 2D map
#   'temporal'         : one segment per player, all features share it
#   'feature'          : one whole feature per player (num_segments collapses to 1)
AXES = ('temporal_feature', 'temporal', 'feature')


def _axis_segments(axis: str, num_segments: int, lf: int):
    ns = 1 if axis == 'feature' else min(num_segments, lf)
    bounds = np.linspace(0, lf, ns + 1, dtype=int)
    seg_len = np.diff(bounds)
    seg_of_frame = np.repeat(np.arange(ns), seg_len)  # (lf,)
    return ns, seg_len, seg_of_frame


def _axis_n_players(axis: str, n_obs_feats: int, ns: int) -> int:
    return {
        'temporal_feature': n_obs_feats * ns,
        'temporal': ns,
        'feature': n_obs_feats
    }[axis]


def _axis_pinned_mask(axis: str, inverse_permutation: np.ndarray, k: int,
                      n_obs_feats: int, ns: int, seg_of_frame: np.ndarray,
                      lf: int) -> np.ndarray:
    """Coalition of size k -> boolean pinned mask over observed (feature,frame) nodes."""
    if axis == 'temporal_feature':
        pinned = inverse_permutation.reshape(n_obs_feats,
                                             ns) < k  # (feat, seg)
        return pinned[:, seg_of_frame].reshape(-1)
    if axis == 'temporal':
        pinned = inverse_permutation < k  # (ns,) segment pinned across all features
        return np.tile(pinned[seg_of_frame], n_obs_feats)
    pinned = inverse_permutation < k  # (n_obs_feats,) whole feature pinned
    return np.repeat(pinned, lf)


def _axis_output(axis: str, contrib: np.ndarray, n_obs_feats: int, ns: int,
                 seg_len: np.ndarray, num_feats: int, lf: int) -> np.ndarray:
    """Shape per-player contributions into the axis's native, model-comparable form.

    temporal_feature -> (num_feats, lf) per-frame density (spread_segments);
    temporal -> (lf,) per-frame density; feature -> (num_feats,). Hazard rows/entry
    are zero (the readout is never a player).
    """
    if axis == 'temporal_feature':
        density = contrib.reshape(n_obs_feats, ns) / seg_len[None, :]
        m = np.repeat(density, seg_len, axis=1)  # (n_obs_feats, lf)
        out = np.zeros((num_feats, lf))
        out[:n_obs_feats] = m
        return out
    if axis == 'temporal':
        return np.repeat(contrib / seg_len, seg_len)  # (lf,) density
    out = np.zeros(num_feats)
    out[:n_obs_feats] = contrib
    return out


def _shapley_axis(make_value_fn: callable, axis: str, num_segments: int,
                  n_obs_feats: int, num_feats: int, lf: int,
                  num_permutations: int) -> np.ndarray:
    """Monte-Carlo Shapley for one game/axis, averaged over permutations.

    ``make_value_fn()`` returns a fresh ``v = value_fn(pinned_mask)`` per
    permutation — the observational estimator uses that to redraw its
    common-random-number seed, so every coalition in a permutation shares
    randomness and a dummy player's marginal is identically zero.

    Permutations are drawn in antithetic pairs (a random order, then its
    reverse), so a player that joins early in one joins late in the other;
    this cancels much of the Monte-Carlo variance from coalition size.
    """
    ns, seg_len, seg_of_frame = _axis_segments(axis, num_segments, lf)
    n_players = _axis_n_players(axis, n_obs_feats, ns)
    acc = np.zeros(n_players)
    perm = None
    for i in range(num_permutations):
        value_fn = make_value_fn()
        perm = (np.random.permutation(n_players)
                if i % 2 == 0 else perm[::-1].copy())
        inv = np.argsort(perm)
        # k = 0..n_players inclusive: the full coalition is needed for the
        # last player's marginal
        v = np.zeros(n_players + 1)
        cache = {}
        for k in range(n_players + 1):
            mask = _axis_pinned_mask(axis, inv, k, n_obs_feats, ns,
                                     seg_of_frame, lf)
            key = mask.tobytes()
            if key not in cache:
                cache[key] = value_fn(mask)
            v[k] = cache[key]
        acc += _contributions_from_values(v, perm)
    return _axis_output(axis, acc / num_permutations, n_obs_feats, ns, seg_len,
                        num_feats, lf)


def _interventional_value(pinned_mask: np.ndarray, base_sig_lf: np.ndarray,
                          bg_sig_lf: np.ndarray, graph, base_signals,
                          base_noise, lf, seq_len, ctx, n_boundary,
                          hazard_calibration_func, hazard_bins,
                          output_type) -> float:
    """Distributional interventional v(S): pinned -> base, absent -> bg draws, avg.

    ``pinned_mask``, ``base_sig_lf`` and ``bg_sig_lf`` cover the window's
    past only; the ``ctx`` context frames before it are not players and stay
    at their base values (the intervention only acts on players)."""
    num_backgrounds = bg_sig_lf.shape[1]
    n_past = ctx + lf
    perturbed = {}
    hazard_idx = None
    for f_idx, f in enumerate(graph.features):
        if f.name == 'Hazard':
            hazard_idx = f_idx
            continue
        rows = slice(f_idx * lf, (f_idx + 1) * lf)  # observed feats come first
        m = pinned_mask[rows][:, None]
        signal = np.empty((seq_len, num_backgrounds))
        signal[:ctx] = base_signals[f_idx, :ctx, None]
        signal[ctx:n_past] = np.where(m, base_sig_lf[rows][:, None],
                                      bg_sig_lf[rows])
        signal[n_past:] = base_signals[f_idx, n_past:seq_len, None]
        perturbed[f.name] = signal
    perturbed['Hazard'] = _hazard_history(base_signals[hazard_idx],
                                          base_noise[hazard_idx], seq_len,
                                          n_boundary, num_backgrounds)
    perturbed = _apply_hazard_rules(graph,
                                    perturbed,
                                    seq_len,
                                    start=n_boundary)
    hazard = hazard_calibration_func(perturbed['Hazard'][n_past:seq_len])
    target = _outputs_from_hazard(hazard, hazard_bins).get_target(
        output_type, hazard_bins)
    return float(np.mean(target))


def _with_context(window_mask: np.ndarray, n_obs_feats: int, lf: int,
                  ctx: int, n_boundary: int) -> np.ndarray:
    """Prefix each feature's window mask (feature-major, ``lf`` frames each)
    with its context frames: the first ``n_boundary`` pinned (the state the
    recursion starts from), the rest free, since they are never players and
    an observer cannot see them."""
    if ctx == 0:
        return window_mask
    prefix = np.zeros((n_obs_feats, ctx), dtype=bool)
    prefix[:, :n_boundary] = True
    return np.concatenate(
        [prefix, window_mask.reshape(n_obs_feats, lf)], axis=1).reshape(-1)


def _make_obs_value_factory(prec,
                            graph,
                            base_signals,
                            base_noise,
                            seq_len,
                            lf,
                            ctx,
                            n_boundary,
                            hazard_calibration_func,
                            hazard_bins,
                            output_type,
                            num_samples,
                            survival_start: int | None = None):
    n_past = prec.lf
    n_obs_feats = len(prec.obs)
    basevec = np.array([
        base_signals[fi][t] for fi in range(n_obs_feats)
        for t in range(n_past)
    ])

    def make():
        xi = np.random.normal(size=(prec.prec.incidence.shape[0], num_samples))

        def value_fn(pinned_mask):
            full_mask = _with_context(pinned_mask, n_obs_feats, lf, ctx,
                                      n_boundary)
            return _coalition_value(prec, full_mask, basevec, xi, graph,
                                    base_signals, base_noise, seq_len,
                                    n_boundary, hazard_calibration_func,
                                    hazard_bins, output_type, num_samples,
                                    survival_start)

        return value_fn

    return make


def _make_interv_value_factory(base_sig_lf, bg_sig_lf, graph, base_signals,
                               base_noise, lf, seq_len, ctx, n_boundary,
                               hazard_calibration_func, hazard_bins,
                               output_type):

    def value_fn(pinned_mask):
        return _interventional_value(pinned_mask, base_sig_lf, bg_sig_lf,
                                     graph, base_signals, base_noise, lf,
                                     seq_len, ctx, n_boundary,
                                     hazard_calibration_func, hazard_bins,
                                     output_type)

    return lambda: value_fn  # background draws are fixed -> CRN across coalitions


def generate_sample_with_importances(
    graph: CausalGraph,
    hazard_calibration_func: callable,
    horizon: int,
    max_sequence_length: int,
    min_sequence_length: int,
    num_permutations: int = 100,
    hazard_bins: np.ndarray | None = None,
    output_type: Literal['expected_time', 'binary'] = 'expected_time',
    min_horizon_cif: float = 0.0,
    num_backgrounds: int = 8,
    num_segments: int | None = None,
    context_frames: int | None = None,
    condition_on_survival: bool = True,
) -> SampleWithSHAP | None:
    """Ground-truth permutation Shapley values over (feature, frame) players.

    One permutation is simulated per iteration. Batching several permutations
    into one rollout does not pay: every phase of an estimator is linear in the
    column count, and a single permutation already spans
    ``num_feats * lf * num_backgrounds`` columns, so there is no per-iteration
    Python overhead left to amortise.

    If the graph has a burn-in, the last ``context_frames`` of it (default:
    all of it, which is exact) are kept before the window. Only window frames
    are players. The first ``graph.max_lag`` context frames are held at their
    base values as the recursions' starting state; the rest are free in the
    observational game (an observer cannot see them) and at base in the
    interventional one. A shorter context is faster; the approximation is
    conditioning on the boundary state, whose influence decays with the
    features' and Hazard's memory.

    ``condition_on_survival``: the observational game conditions on the cell
    being alive at the landmark (as every sample, and every cell a model
    sees, is), by weighting draws by their survival from the start of the
    death-active burn-in (or frame 0 without one). The interventional game
    is a do-operation and is never conditioned.
    """
    ctx = graph.burn_in if context_frames is None else min(
        context_frames, graph.burn_in)
    n_boundary = graph.max_lag if ctx > 0 else 0
    if 0 < ctx <= n_boundary:
        raise ValueError(
            f'context_frames ({ctx}) must exceed the longest rule lag '
            f'({n_boundary}), which is held fixed as the boundary state.')

    base = _generate_base_sample(graph, max_sequence_length, horizon,
                                 min_sequence_length, hazard_calibration_func,
                                 ctx)

    if base is None:
        return None

    lf = base.lf
    n_past = ctx + lf  # past frames in extended coordinates
    seq_len = n_past + horizon
    feat_idx = {f.name: i for i, f in enumerate(graph.features)}

    # Reject degenerate (near-zero in-horizon hazard) samples before doing the
    # expensive permutation work below.
    if min_horizon_cif > 0.0:
        horizon_cif = _realised_horizon_cif(base.signals[feat_idx['Hazard']],
                                            n_past, horizon,
                                            hazard_calibration_func)
        if horizon_cif < min_horizon_cif:
            return None

    num_feats = len(graph.features)

    base_signal_array = base.signals[:, ctx:n_past].reshape(-1)

    # Backgrounds come from the same population as the samples: burnt-in
    # cells alive at frame 0. (num_feats * lf, B), feature-major rows.
    bg = graph.simulate(num_backgrounds,
                        lf,
                        hazard_func=hazard_calibration_func,
                        post_noise=False)
    bg_sig_lf = np.ascontiguousarray(
        np.concatenate([bg[f.name] for f in graph.features], axis=0))

    observed_precision = _build_observed_precision(graph, base.signals, n_past)
    joint_num_segments = (min(num_segments, lf)
                          if num_segments is not None else lf)

    n_obs_feats = sum(f.name != 'Hazard' for f in graph.features)

    make_interv = _make_interv_value_factory(base_signal_array, bg_sig_lf,
                                             graph, base.signals, base.noise,
                                             lf, seq_len, ctx, n_boundary,
                                             hazard_calibration_func,
                                             hazard_bins, output_type)
    interv = {
        axis:
        _shapley_axis(make_interv, axis, joint_num_segments, n_obs_feats,
                      num_feats, lf, num_permutations)
        for axis in AXES
    }

    survival_start = (max(n_boundary, ctx - graph.death_burn_in)
                      if condition_on_survival else None)
    make_obs = _make_obs_value_factory(observed_precision, graph, base.signals,
                                       base.noise, seq_len, lf, ctx,
                                       n_boundary, hazard_calibration_func,
                                       hazard_bins, output_type,
                                       num_backgrounds, survival_start)
    obs = {
        axis:
        _shapley_axis(make_obs, axis, joint_num_segments, n_obs_feats,
                      num_feats, lf, num_permutations)
        for axis in AXES
    }

    obs_boundaries = (np.linspace(0, lf, joint_num_segments + 1, dtype=int)
                      if num_segments is not None else None)

    shap_vals = {}
    for background, axis_results in ((BackgroundEnum.INTERVENTIONAL, interv),
                                     (BackgroundEnum.OBSERVATIONAL, obs)):
        shap_vals[(ExplainerEnum.GROUND_TRUTH, background)] = SHAPResult(
            explainer=ExplainerEnum.GROUND_TRUTH,
            background=background,
            temporal=axis_results['temporal'],
            feature=axis_results['feature'],
            temporal_feature=axis_results['temporal_feature'],
            segment_boundaries=obs_boundaries,
        )

    horizon_hazard = hazard_calibration_func(base.signals[feat_idx['Hazard'],
                                                          n_past:seq_len])

    return SampleWithSHAP(
        feature_vals=base.signals[:, ctx:],
        feature_names=[f.name for f in graph.features],
        landmark_frame=base.lf,
        death_frame=base.death_frame,
        noise_vals=base.noise[:, ctx:],
        horizon_hazard=horizon_hazard,
        shap_vals=shap_vals,
        context_vals=base.signals[:, :ctx] if ctx > 0 else None,
    )
