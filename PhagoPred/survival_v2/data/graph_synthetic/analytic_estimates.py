from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, fields, field
from pathlib import Path

import numpy as np
import h5py
from tqdm import tqdm
from scipy import sparse as _sparse
from scipy.sparse.linalg import splu as _splu

from .graph import CausalGraph


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
    signals: dict
    noise: dict
    lf: int
    death_frame: int


@dataclass
class sampleWithImportances:
    base_signals: dict
    base_noise: dict
    landmark_frame: int
    death_frame: int | None
    segment_boundaries: np.ndarray | None = None
    # ``*_importances`` are the joint (feature, frame) 2D maps. The temporal and
    # feature marginals are *separate* Shapley games (matching KernelSHAP), not
    # projections of the 2D map, stored so plotting shows the faithful per-axis
    # value. temporal: (lf,) per-frame density; feature: (num_feats,).
    interventional_importances: np.ndarray | None = None
    observational_importances: np.ndarray | None = None
    interventional_temporal: np.ndarray | None = None
    observational_temporal: np.ndarray | None = None
    interventional_feature: np.ndarray | None = None
    observational_feature: np.ndarray | None = None
    # Calibrated hazard along the realised trajectory over [lf, lf + horizon).
    # The base Hazard *signal* is already in base_signals, but the calibration
    # func that turns it into a probability is not, so store the result.
    horizon_hazard: np.ndarray | None = None

    def to_h5(self, h5_path: Path, sample_idx: int | None = None) -> None:
        mode = 'a' if sample_idx is not None else 'w'
        with h5py.File(h5_path, mode) as f:
            if sample_idx is not None:
                name = str(sample_idx)
                if name in f:
                    del f[name]
                group = f.create_group(name)
            else:
                group = f

            group.attrs['Features'] = list(self.base_signals.keys())
            group.attrs['Landmark Frame'] = self.landmark_frame
            if self.death_frame is not None:
                group.attrs['Death Frame'] = self.death_frame
            if self.segment_boundaries is not None:
                group.attrs['Segment Boundaries'] = self.segment_boundaries
            group.create_dataset('Noise',
                                 data=np.stack(list(self.base_noise.values()),
                                               axis=0),
                                 dtype=float)
            group.create_dataset('Signals',
                                 data=np.stack(list(
                                     self.base_signals.values()),
                                               axis=0),
                                 dtype=float)
            for name, mp, temporal, feat in (
                ('Interventional', self.interventional_importances,
                 self.interventional_temporal, self.interventional_feature),
                ('Observational', self.observational_importances,
                 self.observational_temporal, self.observational_feature)):
                if mp is None:
                    continue
                ds = group.create_dataset(name, data=mp, dtype=float)
                if temporal is not None:
                    ds.attrs['Temporal'] = temporal  # (lf,) separate game
                if feat is not None:
                    ds.attrs['Feature'] = feat  # (num_feats,) separate game
            if self.horizon_hazard is not None:
                group.create_dataset('Horizon Hazard',
                                     data=self.horizon_hazard,
                                     dtype=float)

    @classmethod
    def from_h5(cls, h5_path: Path, sample_idx: int | None = None):
        with h5py.File(h5_path, 'r') as f:
            group = f[str(sample_idx)] if sample_idx is not None else f

            features = [
                n.decode() if isinstance(n, bytes) else str(n)
                for n in group.attrs['Features']
            ]
            landmark_frame = int(group.attrs['Landmark Frame'])
            death_frame = group.attrs.get('Death Frame', None)
            segment_boundaries = group.attrs.get('Segment Boundaries', None)

            noise = group['Noise'][:]
            base_noise = {name: noise[i] for i, name in enumerate(features)}

            signals = group['Signals'][:]
            base_signals = {
                name: signals[i]
                for i, name in enumerate(features)
            }

            def _read(name):
                if name not in group:
                    return None, None, None
                ds = group[name]
                return (ds[:], ds.attrs.get('Temporal', None),
                        ds.attrs.get('Feature', None))

            (interventional_importances, interventional_temporal,
             interventional_feature) = _read('Interventional')
            (observational_importances, observational_temporal,
             observational_feature) = _read('Observational')
            # Absent in files written before horizon_hazard was stored.
            horizon_hazard = (group['Horizon Hazard'][:]
                              if 'Horizon Hazard' in group else None)

        return cls(
            base_signals,
            base_noise,
            landmark_frame,
            death_frame,
            segment_boundaries,
            interventional_importances,
            observational_importances,
            interventional_temporal,
            observational_temporal,
            interventional_feature,
            observational_feature,
            horizon_hazard,
        )


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    sf = np.cumprod(1.0 - hazards, axis=0)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def _apply_rules(graph: CausalGraph,
                 noise: dict[str, np.ndarray],
                 time_steps: int,
                 copy: bool = True) -> dict[str, np.ndarray]:
    # == OPTIONALLY ADD POST_NOISE===
    # copy=False propagates in place; only safe when the caller built ``noise``
    # solely for this rollout and does not need the innovations afterwards.
    signals = copy_signals(noise) if copy else noise
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
    return signals


def _apply_rules_obs(
    graph: CausalGraph,
    signals: dict[str, np.ndarray],  # (seq_len, n_cols), init to innovations
    past_noise: dict[str, np.ndarray],  # (lf, n_cols) innovations, no Hazard
    fixed_past: dict[str, np.ndarray],  # (lf, n_cols) pinned signal, no Hazard
    time_steps: int,
    lf: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Propagate the graph, pinning every past node to its coalition signal.

    ``signals`` is mutated in place. Every past (feature, frame) node is pinned —
    to the base signal if the player is in the coalition, otherwise to a
    background draw — so there is no "unset" past node and nothing to pin after
    ``lf``. The pinning is therefore an unconditional assignment for ``t < lf``
    and skipped entirely beyond it. Hazard has no pre-noise density of interest
    and is left to propagate, carrying the pinned past to the readout.
    """
    # === AGAIN OPTIONALLY ADD POST_NOISE ===
    logw = 0.0
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
        if t >= lf:
            continue
        for f in graph.features:
            if f.name == 'Hazard':
                continue
            innovation = past_noise[f.name][t]
            expr_part = signals[f.name][t] - innovation
            fixed = fixed_past[f.name][t]
            logw = logw + (f.pre_noise.log_prob(fixed - expr_part) -
                           f.pre_noise.log_prob(innovation))
            signals[f.name][t] = fixed
    return signals, logw


def _apply_rules_pin_present(
    graph: CausalGraph,
    signals: dict[str, np.ndarray],  # (seq_len, n_cols), init to innovations
    past_noise: dict[str, np.ndarray],  # (lf, n_cols) innovations, no Hazard
    base_past: dict[str, np.ndarray],  # (lf, 1) base signal, no Hazard
    present: dict[str, np.ndarray],  # (lf, n_cols) bool, no Hazard
    time_steps: int,
    lf: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Propagate the graph, pinning only the *present* past nodes to base.

    ``signals`` is mutated in place. A present (feature, frame) node is pinned to
    its base value and pays the log-density of the innovation that pinning
    requires. An absent node is left exactly as the rules produced it — it keeps
    propagating from its own background innovation, is never pinned, and carries
    no weight.

    Every column is therefore a path the dynamics can actually produce: it drifts
    off base between pins and is snapped back at each one. Contrast
    ``_apply_rules_obs``, which pins absent nodes onto an independent background
    *signal* — a splice reachable only via innovations many sigma into the tail,
    so its weights span tens of thousands of log units and collapse onto a single
    background draw.

    Absent nodes carrying no weight has a second consequence worth keeping: a
    feature that never reaches the readout can only enter v(S) through its own
    pinned nodes, so it cannot steer which background draw dominates.

    Hazard has no pre-noise density of interest and is left to propagate,
    carrying the pinned past to the readout.
    """
    logw = 0.0
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
        if t >= lf:
            continue
        for f in graph.features:
            if f.name == 'Hazard':
                continue
            # Every rule reads strictly lagged values, so by the time frame t is
            # written its drift is fully determined and the innovation separates
            # out cleanly — whichever innovation the column happens to hold.
            drift = signals[f.name][t] - past_noise[f.name][t]
            here = present[f.name][t]
            base_here = base_past[f.name][t]
            logw = logw + np.where(
                here, f.pre_noise.log_prob(base_here - drift), 0.0)
            signals[f.name][t] = np.where(here, base_here, signals[f.name][t])
    return signals, logw


def _self_coeffs(graph: CausalGraph) -> dict[str, float]:
    """Linearised lag-1 self-coefficient of each observed feature.

    The bridge geometry treats a free run of one feature as an AR(1) process
    ``x[t] = a x[t-1] + eps`` and aims it at the next pinned frame. ``a`` is that
    self-coefficient, read straight from the rules via their analytic ``partial``
    (exact for ``AutoCorrelationRule``; the tangent slope at 0 for anything
    nonlinear). Cross-feature terms are not part of ``a`` — they enter each step
    as a known drift, and their *backward* influence is the job of the (not yet
    wired) cross-feature message.
    """
    coeffs = {}
    for f in graph.features:
        if f.name == 'Hazard':
            continue
        a = 0.0
        for rule in graph.rules:
            if rule.target != f.name:
                continue
            ctx = {
                (n, lag): 0.0
                for n, lags in rule.inputs.items()
                for lag in lags
            }
            a += rule.expr.partial(f.name, 1, ctx)
        coeffs[f.name] = float(a)
    return coeffs


def _pre_scale(noise) -> float:
    """Std/scale of a feature's pre-noise, for the Gaussian bridge variance.

    Only the proposal uses this Gaussian; the weight scores the true pre-noise
    density, so a mismatched scale (or a non-Gaussian pre-noise) costs ESS, never
    correctness.
    """
    return float(getattr(noise, 'sigma', getattr(noise, 'scale', 1.0)))


def _lookahead(present_f: np.ndarray, base_sig_f: np.ndarray, lf: int):
    """Per (frame, column): the next pinned frame strictly after t, and its base.

    ``present_f`` is (lf, n_cols) bool. Returns ``k`` (frames to the next pin, 0
    where there is none ahead) and ``base_next`` (the base signal at that pinned
    frame, 0 where none) — the two quantities the AR(1) bridge aims at.
    """
    n_cols = present_f.shape[1]
    nxt = np.full((lf, n_cols), -1, dtype=np.int64)
    last = np.full(n_cols, -1, dtype=np.int64)
    for t in range(lf - 1, -1, -1):
        nxt[t] = last  # nearest pinned frame with index > t
        last = np.where(present_f[t], t, last)
    frames = np.arange(lf)[:, None]
    k = np.where(nxt >= 0, nxt - frames, 0)
    base_next = np.where(nxt >= 0, base_sig_f[np.clip(nxt, 0, lf - 1)], 0.0)
    return k, base_next


def _apply_rules_bridge(
    graph: CausalGraph,
    signals: dict[str,
                  np.ndarray],  # (seq_len, n_cols), past init to innovations
    past_noise: dict[str, np.ndarray],  # (lf, n_cols) innovations, no Hazard
    base_past: dict[str, np.ndarray],  # (lf, 1) base signal, no Hazard
    present: dict[str, np.ndarray],  # (lf, n_cols) bool, no Hazard
    k_look: dict[str,
                 np.ndarray],  # (lf, n_cols) frames to next pin, no Hazard
    base_next: dict[str,
                    np.ndarray],  # (lf, n_cols) base @ next pin, no Hazard
    a_coeffs: dict[str, float],
    scales: dict[str, float],
    time_steps: int,
    lf: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Like ``_apply_rules_pin_present`` but *sample* absent nodes from a bridge.

    Three cases per past (feature, frame, column):

    * **present** -> pin to base, pay ``log p_eps(base - drift)`` (unchanged).
    * **absent, a pin still ahead** -> draw from the AR(1) *bridge* proposal
      N(mean, var) aimed at the next pinned frame, and pay the exact importance
      ratio ``log p_eps(z - drift) - log q(z)``. Over a free run these ratios
      telescope with the closing pin to ``p(base_next | base_prev)`` — a constant
      across columns — so the weights stop fanning out and ESS stays ~B.
    * **absent, no pin ahead** -> draw from the prior transition, weight 0
      (proposal == prior); nothing downstream constrains it.

    The bridge is the two-Gaussian product of the prior transition
    ``N(drift, v0)`` with the k-step lookahead ``N(a^k x, v0 S_k)`` to the pinned
    value — precisions add. It is *exact* when a feature's free run is a pure
    AR(1) between pins (no cross-feature edges among observed features), which
    holds for base_linear / base_multiplicative / base_ratio.

    CROSS-FEATURE SEAM: a pinned *descendant in another feature* would contribute
    a second lookahead message here; multiplied into (mean, var) it is another
    Gaussian factor (linearised via EKS/UKS for a nonlinear edge). It is omitted
    for now, so base_chain's A->B->C coupling is proposed self-chain-only: still
    unbiased (the weight is exact), just lower ESS than the exact smoother.
    """
    logw = 0.0
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
        if t >= lf:
            continue
        for f in graph.features:
            name = f.name
            if name == 'Hazard':
                continue
            # Strictly-lagged rules -> drift at t is fully determined; the
            # innovation separates out whatever value the column happens to hold.
            drift = signals[name][t] - past_noise[name][t]  # (n_cols,)
            here = present[name][t]  # (n_cols,) bool
            base_here = base_past[name][t]  # (1,) -> broadcasts
            kk = k_look[name][t]  # (n_cols,) int, 0 where no pin ahead
            has = (~here) & (kk > 0)

            a = a_coeffs[name]
            v0 = scales[name]**2
            a2 = a * a
            with np.errstate(over='ignore', invalid='ignore'):
                ak = np.where(kk > 0, a**kk, 1.0)
                if abs(1.0 - a2) < 1e-9:
                    sk = np.maximum(kk,
                                    1)  # limit of the geometric sum as a->1
                else:
                    sk = np.where(kk > 0, (1.0 - a2**kk) / (1.0 - a2), 1.0)
            skv = v0 * sk  # variance the AR(1) accumulates over k steps
            prec = 1.0 / v0 + (ak * ak) / skv
            mean = (drift / v0 + ak * base_next[name][t] / skv) / prec
            var = 1.0 / prec

            z_bridge = mean + np.random.normal(size=drift.shape) * np.sqrt(var)
            log_q = (-0.5 * (z_bridge - mean)**2 / var -
                     0.5 * np.log(2 * np.pi * var))
            lp_bridge = f.pre_noise.log_prob(z_bridge - drift) - log_q

            z_prior = drift + np.random.normal(size=drift.shape) * scales[name]
            lp_present = f.pre_noise.log_prob(base_here - drift)

            logw = logw + np.where(here, lp_present,
                                   np.where(has, lp_bridge, 0.0))
            signals[name][t] = np.where(here, base_here,
                                        np.where(has, z_bridge, z_prior))
    return signals, logw


def _apply_hazard_rules(graph: CausalGraph, signals: dict[str, np.ndarray],
                        time_steps: int) -> dict:
    hazard_rules = [r for r in graph.rules if r.target == 'Hazard']
    for t in range(time_steps):
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
    # return {'hazard': hazard, 'cdf': cdf, 'pmf': pmf}


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


def copy_signals(signals: dict) -> dict:
    return {k: v.copy() for k, v in signals.items()}


def _generate_base_sample(
        graph: CausalGraph, seq_len: int, horizon: int, min_seq_len: int,
        hazard_calibration_func: callable) -> baseSample | None:
    base_noise = {
        f.name: f.generate_signal(seq_len + horizon)
        for f in graph.features
    }
    base_signals = _apply_rules(graph, base_noise, seq_len + horizon)
    landmark_sample = _landmark_sample(
        base_signals,
        hazard_calibration_func,
        seq_len,
        min_seq_len,
    )
    if landmark_sample is None:
        return None

    lf, death_frame = landmark_sample

    return baseSample(base_signals, base_noise, lf, death_frame)


def _generate_mean_background_sample(graph: CausalGraph,
                                     seq_len: int,
                                     num_samples: int = 100):
    draws = [{
        f.name: f.generate_signal(seq_len)
        for f in graph.features
    } for _ in tqdm(range(num_samples), 'Estimating mean background')]
    noise = {
        f.name: np.stack([d[f.name] for d in draws], axis=1)
        for f in graph.features
    }  # (seq_len, num_samples)
    signals = _apply_rules(graph, noise, seq_len, copy=False)
    return {f.name: signals[f.name].mean(axis=1) for f in graph.features}


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


def _coalition_membership(inverse_permutation: np.ndarray,
                          n_players: int) -> np.ndarray:
    """``present[i, k]``: player i is in the size-k coalition.

    ``inverse_permutation[i]`` is player i's position in the permutation, and
    coalition sizes run 0 .. n_players-1. The result is *player-major*: players
    are laid out feature-major down axis 0, so one feature's players are a
    contiguous row slice. That lets each feature's coalition block be written
    straight into its (seq_len, n_cols) signal buffer, with no transpose and no
    fancy-index scatter.
    """
    return inverse_permutation[:, None] < np.arange(n_players)[None, :]


def _coalition_blocks(present_rows: np.ndarray, base_row: np.ndarray,
                      bg_rows: np.ndarray, n_cols: int) -> np.ndarray:
    """One feature's past window: base value where present, else a background draw.

    ``present_rows`` (lf, n_coal), ``base_row`` (lf,), ``bg_rows`` (lf, B).
    Returns (lf, n_cols) with column = coalition * B + draw.
    """
    return np.where(present_rows[:, :, None], base_row[:, None, None],
                    bg_rows[:, None, :]).reshape(len(base_row), n_cols)


def _contributions_from_values(values: np.ndarray,
                               permutation_order: np.ndarray) -> np.ndarray:
    """Marginals of ``values`` (n_coal,) scattered back to player order."""
    marginals = np.diff(values)
    contributions = np.zeros(permutation_order.shape[0])
    contributions[permutation_order[:-1]] = marginals
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
class _ObservedPrecision:
    obs: list  # observed (non-Hazard) feature names, in graph order
    n_obs: int  # number of observed nodes = len(obs) * lf
    incidence: object  # scipy csc (n_eq, n_obs): one linearised eps-equation per row
    sqrt_iv: np.ndarray  # (n_eq,) sqrt(1/pre-noise-var) per equation
    Q: object  # scipy csc (n_obs, n_obs) = incidenceᵀ diag(iv) incidence + ridge
    is_linear: bool
    lf: int


def _observed_is_linear(graph: CausalGraph) -> bool:
    """True iff no rule targeting an *observed* feature is nonlinear.

    Nonlinearity at the Hazard sink does not count — Hazard is never sampled, so
    the conditional over observed nodes stays linear-Gaussian and needs no
    importance reweighting.
    """
    from . import rules as R
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
                              base_signals: dict,
                              lf: int,
                              ridge: float = 1e-8) -> _ObservedPrecision:
    """Linear-Gaussian precision over observed past nodes, once per base sample.

    Each observed node (f, t) contributes an innovation equation
    ``eps = x[f,t] - sum_parents coef * x[parent, t-lag]``.  For a nonlinear rule
    ``coef`` is the EKF Jacobian (``Expr.partial``) at the *base* trajectory, so
    the precision is the linearised proposal; the exact rule is used later in the
    importance weight.  Node column == full-player index because the observed
    (non-Hazard) features come first in ``graph.features``.
    """
    obs = [f.name for f in graph.features if f.name != 'Hazard']
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
                    (nm, lg): base_signals[nm][t - lg]
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
    sqrt_iv = np.asarray(sqrt_iv)
    lam = _sparse.diags(sqrt_iv**2)
    Q = (incidence.T @ lam @ incidence + ridge * _sparse.eye(n_obs)).tocsc()
    return _ObservedPrecision(obs, n_obs, incidence, sqrt_iv, Q,
                              _observed_is_linear(graph), lf)


def _col_to_ft(cols: np.ndarray, obs: list, lf: int):
    """Vectorised inverse of the (feature, frame) -> column map."""
    return cols // lf, cols % lf  # feature index, frame


def _propagate_and_target(graph, past, base_noise, lf, seq_len,
                          hazard_calibration_func, hazard_bins, output_type,
                          B):
    """Fill observed past with ``past`` samples, propagate the horizon, read target."""
    sig = {f.name: np.zeros((seq_len, B)) for f in graph.features}
    for f in graph.features:
        if f.name in past:
            sig[f.name][:lf] = past[f.name]
        sig[f.name][lf:] = base_noise[f.name][lf:seq_len, None]
    for t in range(lf, seq_len):
        for rule in graph.rules:
            rule.apply_step(sig, t)
    hazard = hazard_calibration_func(sig['Hazard'][lf:seq_len])
    return _outputs_from_hazard(hazard, hazard_bins).get_target(
        output_type, hazard_bins)  # (B,)


def _true_log_prior(graph, past, prec: _ObservedPrecision,
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


def _coalition_value(prec: _ObservedPrecision, pinned_mask: np.ndarray,
                     basevec: np.ndarray, xi: np.ndarray, graph, base_signals,
                     base_noise, seq_len, hazard_calibration_func, hazard_bins,
                     output_type, B) -> float:
    """v(S) = E[target | observed pinned = base] via joint-conditional samples."""
    lf, obs = prec.lf, prec.obs
    F = np.where(~pinned_mask)[0]
    P = np.where(pinned_mask)[0]
    past = {f: np.repeat(base_signals[f][:lf, None], B, axis=1) for f in obs}
    logw = np.zeros(B)
    if len(F) > 0:
        Qcsr = prec.Q.tocsr()
        Qff = Qcsr[F][:, F].tocsc()
        lu = _splu(Qff)
        rhs = -(Qcsr[F][:, P] @ basevec[P]) if len(P) else np.zeros(len(F))
        w = prec.incidence[:, F].T @ (prec.sqrt_iv[:, None] * xi
                                      )  # ~N(0, Q_FF)
        Z = lu.solve(rhs[:, None] + w)  # (n_free, B) ~ N(mean, Q_FF^{-1})
        fis, ts = _col_to_ft(F, obs, lf)
        for fi in range(len(obs)):
            sel = fis == fi
            if sel.any():
                past[obs[fi]][ts[sel]] = Z[np.where(sel)[0]]
        if not prec.is_linear:  # reweight the EKF-linearised proposal
            mean = lu.solve(rhs)
            d = Z - mean[:, None]
            logdet = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
            logq = (0.5 * logdet - 0.5 * np.einsum('ib,ib->b', d, Qff @ d) -
                    0.5 * len(F) * np.log(2 * np.pi))
            logw = _true_log_prior(graph, past, prec, B) - logq
    tgt = _propagate_and_target(graph, past, base_noise, lf, seq_len,
                                hazard_calibration_func, hazard_bins,
                                output_type, B)
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
    return {'temporal_feature': n_obs_feats * ns,
            'temporal': ns,
            'feature': n_obs_feats}[axis]


def _axis_pinned_mask(axis: str, inverse_permutation: np.ndarray, k: int,
                      n_obs_feats: int, ns: int, seg_of_frame: np.ndarray,
                      lf: int) -> np.ndarray:
    """Coalition of size k -> boolean pinned mask over observed (feature,frame) nodes."""
    if axis == 'temporal_feature':
        pinned = inverse_permutation.reshape(n_obs_feats, ns) < k  # (feat, seg)
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


def _shapley_axis(make_value_fn, axis: str, num_segments: int,
                  n_obs_feats: int, num_feats: int, lf: int,
                  num_permutations: int) -> np.ndarray:
    """Monte-Carlo Shapley for one game/axis, averaged over permutations.

    ``make_value_fn()`` returns a fresh ``v = value_fn(pinned_mask)`` per
    permutation — the observational estimator uses that to redraw its
    common-random-number seed, so every coalition in a permutation shares
    randomness and a dummy player's marginal is identically zero.
    """
    ns, seg_len, seg_of_frame = _axis_segments(axis, num_segments, lf)
    n_players = _axis_n_players(axis, n_obs_feats, ns)
    acc = np.zeros(n_players)
    for _ in range(num_permutations):
        value_fn = make_value_fn()
        perm = np.random.permutation(n_players)
        inv = np.argsort(perm)
        v = np.zeros(n_players)
        cache = {}
        for k in range(n_players):
            mask = _axis_pinned_mask(axis, inv, k, n_obs_feats, ns, seg_of_frame,
                                     lf)
            key = mask.tobytes()
            if key not in cache:
                cache[key] = value_fn(mask)
            v[k] = cache[key]
        acc += _contributions_from_values(v, perm)
    return _axis_output(axis, acc / num_permutations, n_obs_feats, ns, seg_len,
                        num_feats, lf)


def _interventional_value(pinned_mask: np.ndarray, base_sig_lf: np.ndarray,
                          bg_sig_lf: np.ndarray, graph, base_signals, base_noise,
                          lf, seq_len, hazard_calibration_func, hazard_bins,
                          output_type) -> float:
    """Distributional interventional v(S): pinned -> base, absent -> bg draws, avg."""
    num_backgrounds = bg_sig_lf.shape[1]
    perturbed = {}
    for f_idx, f in enumerate(graph.features):
        if f.name == 'Hazard':
            continue
        rows = slice(f_idx * lf, (f_idx + 1) * lf)  # observed feats come first
        m = pinned_mask[rows][:, None]
        past = np.where(m, base_sig_lf[rows][:, None], bg_sig_lf[rows])
        signal = np.empty((seq_len, num_backgrounds))
        signal[:lf] = past
        signal[lf:] = base_signals[f.name][lf:seq_len, None]
        perturbed[f.name] = signal
    perturbed['Hazard'] = np.broadcast_to(
        base_noise['Hazard'][:seq_len, None],
        (seq_len, num_backgrounds)).copy()
    perturbed = _apply_hazard_rules(graph, perturbed, seq_len)
    hazard = hazard_calibration_func(perturbed['Hazard'][lf:seq_len])
    target = _outputs_from_hazard(hazard, hazard_bins).get_target(
        output_type, hazard_bins)
    return float(np.mean(target))


def _make_obs_value_factory(prec, graph, base_signals, base_noise, seq_len,
                            hazard_calibration_func, hazard_bins, output_type,
                            num_samples):
    lf = prec.lf
    basevec = np.array(
        [base_signals[f][t] for f in prec.obs for t in range(lf)])

    def make():
        xi = np.random.normal(size=(prec.incidence.shape[0], num_samples))

        def value_fn(pinned_mask):
            return _coalition_value(prec, pinned_mask, basevec, xi, graph,
                                    base_signals, base_noise, seq_len,
                                    hazard_calibration_func, hazard_bins,
                                    output_type, num_samples)

        return value_fn

    return make


def _make_interv_value_factory(base_sig_lf, bg_sig_lf, graph, base_signals,
                               base_noise, lf, seq_len, hazard_calibration_func,
                               hazard_bins, output_type):
    def value_fn(pinned_mask):
        return _interventional_value(pinned_mask, base_sig_lf, bg_sig_lf, graph,
                                     base_signals, base_noise, lf, seq_len,
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
    compute_noise_observational: bool = False,
    observational_method: Literal['bridge', 'joint'] = 'joint',
    num_segments: int | None = None,
) -> sampleWithImportances:
    """Ground-truth permutation Shapley values over (feature, frame) players.

    One permutation is simulated per iteration. Batching several permutations
    into one rollout does not pay: every phase of an estimator is linear in the
    column count, and a single permutation already spans
    ``num_feats * lf * num_backgrounds`` columns, so there is no per-iteration
    Python overhead left to amortise.

    ``_noise_observational`` is a comparison-only baseline whose result is not
    stored on the returned sample, so it is skipped unless
    ``compute_noise_observational`` is set. It consumes no randomness, so
    enabling it does not perturb the other two estimators.
    """

    base = _generate_base_sample(graph, max_sequence_length, horizon,
                                 min_sequence_length, hazard_calibration_func)

    if base is None:
        return None

    lf = base.lf
    seq_len = lf + horizon

    # Reject degenerate (near-zero in-horizon hazard) samples before doing the
    # expensive permutation work below.
    if min_horizon_cif > 0.0:
        horizon_cif = _realised_horizon_cif(base.signals['Hazard'], lf,
                                            horizon, hazard_calibration_func)
        if horizon_cif < min_horizon_cif:
            return None

    num_feats = len(graph.features)
    total_num_feats = num_feats * lf
    total_feats = np.arange(total_num_feats)
    base_signal_array = np.concatenate(
        [base.signals[f.name][:lf] for f in graph.features])

    # Shared background draws (noise + propagated signal), built once and reused
    # across permutations so every estimator sees identical backgrounds.
    # Transposed to player-major so estimators can slice one feature's rows.
    bg_noise_lf = np.empty((num_backgrounds, total_num_feats))
    bg_sig_lf = np.empty((num_backgrounds, total_num_feats))
    for b in range(num_backgrounds):
        bg_noise = {f.name: f.generate_signal(lf) for f in graph.features}
        bg_sig = _apply_rules(graph, bg_noise, lf)
        bg_noise_lf[b] = np.concatenate(
            [bg_noise[f.name][:lf] for f in graph.features])
        bg_sig_lf[b] = np.concatenate(
            [bg_sig[f.name][:lf] for f in graph.features])
    bg_noise_lf = np.ascontiguousarray(bg_noise_lf.T)  # (n_players, B)
    bg_sig_lf = np.ascontiguousarray(bg_sig_lf.T)

    # Joint smoother needs the linear-Gaussian precision over observed nodes,
    # built once from the base trajectory (EKF-linearised for nonlinear edges).
    # Players are (feature, segment); num_segments=None means one per frame.
    observed_precision = (_build_observed_precision(graph, base.signals, lf)
                          if observational_method == 'joint' else None)
    joint_num_segments = (min(num_segments, lf)
                          if num_segments is not None else lf)

    n_obs_feats = sum(f.name != 'Hazard' for f in graph.features)

    # Each estimator is computed as three *separate* Shapley games (temporal,
    # feature, and the joint 2D map) matching KernelSHAP's masking granularities,
    # so every axis is directly comparable rather than a projection of the 2D map.
    make_interv = _make_interv_value_factory(base_signal_array, bg_sig_lf, graph,
                                             base.signals, base.noise, lf,
                                             seq_len, hazard_calibration_func,
                                             hazard_bins, output_type)
    interv = {
        axis: _shapley_axis(make_interv, axis, joint_num_segments, n_obs_feats,
                            num_feats, lf, num_permutations)
        for axis in tqdm(AXES, desc='interventional axes')
    }

    if observational_method == 'joint':
        make_obs = _make_obs_value_factory(observed_precision, graph,
                                           base.signals, base.noise, seq_len,
                                           hazard_calibration_func, hazard_bins,
                                           output_type, num_backgrounds)
        obs = {
            axis: _shapley_axis(make_obs, axis, joint_num_segments, n_obs_feats,
                                num_feats, lf, num_permutations)
            for axis in tqdm(AXES, desc='observational axes')
        }
    else:  # legacy bridge: joint 2D map only; marginals left to projection
        acc = np.zeros(total_num_feats)
        for _ in tqdm(range(num_permutations), desc='observational (bridge)'):
            perm = np.random.permutation(total_feats)
            acc += _observational(num_feats, perm, np.argsort(perm), graph,
                                  base.signals, base.noise, bg_noise_lf, lf,
                                  seq_len, hazard_calibration_func, hazard_bins,
                                  output_type)
        obs = {
            'temporal_feature': (acc / num_permutations).reshape(num_feats, lf),
            'temporal': None,
            'feature': None,
        }

    obs_boundaries = (np.linspace(0, lf, joint_num_segments + 1, dtype=int)
                      if num_segments is not None else None)

    return sampleWithImportances(
        base_signals=base.signals,
        base_noise=base.noise,
        landmark_frame=base.lf,
        death_frame=base.death_frame,
        segment_boundaries=obs_boundaries,
        interventional_importances=interv['temporal_feature'],
        observational_importances=obs['temporal_feature'],
        interventional_temporal=interv['temporal'],
        observational_temporal=obs['temporal'],
        interventional_feature=interv['feature'],
        observational_feature=obs['feature'],
        horizon_hazard=hazard_calibration_func(
            base.signals['Hazard'][lf:seq_len]),
    )


def _observational(
    num_feats: int,
    permutation_order: np.ndarray,  # (n_players,)
    inverse_permutation: np.ndarray,  # (n_players,)
    graph: CausalGraph,
    base_signals: dict[str, np.ndarray],
    base_noise: dict[str, np.ndarray],
    bg_noise_lf: np.ndarray,  # (n_players, num_backgrounds)
    lf: int,
    seq_len: int,
    hazard_calibration_func: callable,
    hazard_bins: np.ndarray | None,
    output_type: Literal['expected_time', 'binary'],
) -> np.ndarray:
    """Observational Shapley values: pin the coalition, bridge the rest.

    A present past player is pinned to its base value and pays the log-density of
    the innovation that pinning requires. An absent player is *sampled from a
    bridge* aimed at its feature's next pinned frame (``_apply_rules_bridge``),
    paying the exact importance ratio; where no pin lies ahead it just propagates
    from the prior, unweighted. v(S) is a self-normalised average over the shared
    ``bg_noise_lf`` draws, which the caller builds once — so coalitions along a
    permutation see identical innovations and consecutive v(S) share random
    numbers, which is what keeps the Shapley differences estimable.

    The bridge is what makes the per-column weights telescope to a constant, so
    ESS stays ~B instead of collapsing as the coalition fills (the failure of the
    plain pin-present scheme). It is exact where a feature's free run is pure
    AR(1) between pins — base_linear / base_multiplicative / base_ratio. For
    base_chain's linear A->B->C coupling, and any nonlinear cross-feature rule,
    the self-chain bridge is only a proposal (still unbiased via the exact
    weight, lower ESS); closing that gap is the cross-feature EKS/UKS message
    flagged in ``_apply_rules_bridge``.
    """
    n_players = num_feats * lf  # players = (feature, frame) over [0, lf)
    n_coal = n_players  # coalition sizes 0 .. n_players-1 (matches others)
    num_backgrounds = bg_noise_lf.shape[1]
    features = graph.features
    n_cols = n_coal * num_backgrounds  # column = coalition * B + draw

    base_sig_lf = np.concatenate([base_signals[f.name][:lf] for f in features])
    base_noise_lf = np.concatenate([base_noise[f.name][:lf] for f in features])
    a_coeffs = _self_coeffs(graph)
    scales = {f.name: _pre_scale(f.pre_noise) for f in features}

    # Membership (player-major): player i is present in coalition k iff its
    # permuted position is < k.
    present_players = _coalition_membership(inverse_permutation,
                                            n_players)  # (n_players, n_coal)

    # Fill each feature's buffers straight from its contiguous slice of players.
    signals, past_noise, base_past, present = {}, {}, {}, {}
    k_look, base_next = {}, {}
    for f_idx, f in enumerate(features):
        rows = slice(f_idx * lf, (f_idx + 1) * lf)
        innovations = _coalition_blocks(present_players[rows],
                                        base_noise_lf[rows], bg_noise_lf[rows],
                                        n_cols)
        signal = np.empty((seq_len, n_cols))
        signal[:lf] = innovations
        signal[lf:] = base_noise[f.name][lf:seq_len, None]
        signals[f.name] = signal
        if f.name == 'Hazard':
            continue  # propagates freely; carries no pinned value or weight
        past_noise[f.name] = innovations
        base_past[f.name] = base_sig_lf[rows][:, None]  # (lf, 1), broadcasts
        # (lf, n_cols), matching _coalition_blocks' column = coalition * B + draw
        present[f.name] = np.repeat(present_players[rows],
                                    num_backgrounds,
                                    axis=1)
        k_look[f.name], base_next[f.name] = _lookahead(
            present[f.name], base_signals[f.name][:lf], lf)

    perturbed_signals, logw = _apply_rules_bridge(graph, signals, past_noise,
                                                  base_past, present, k_look,
                                                  base_next, a_coeffs, scales,
                                                  seq_len, lf)
    hazard = hazard_calibration_func(perturbed_signals['Hazard'][lf:seq_len])
    sample_outputs = _outputs_from_hazard(hazard, hazard_bins)
    target = sample_outputs.get_target(output_type, hazard_bins)  # (n_cols,)

    # Self-normalised importance average over background draws -> v(S).
    logw = np.reshape(logw, (n_coal, num_backgrounds))
    target = np.reshape(target, (n_coal, num_backgrounds))
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    v = (w * target).sum(axis=1) / w.sum(axis=1)  # (n_coal,)

    return _contributions_from_values(v, permutation_order)


def _old_observational(
    num_feats: int,
    permutation_order: np.ndarray,  # (n_players,)
    inverse_permutation: np.ndarray,  # (n_players,)
    graph: CausalGraph,
    base_signals: dict[str, np.ndarray],
    base_noise: dict[str, np.ndarray],
    bg_noise_lf: np.ndarray,  # (n_players, num_backgrounds)
    bg_sig_lf: np.ndarray,  # (n_players, num_backgrounds)
    lf: int,
    seq_len: int,
    hazard_calibration_func: callable,
    hazard_bins: np.ndarray | None,
    output_type: Literal['expected_time', 'binary'],
) -> np.ndarray:
    """Observational Shapley values via signal-pinning + noise reweighting.
    """
    n_players = num_feats * lf  # players = (feature, frame) over [0, lf)
    n_coal = n_players  # coalition sizes 0 .. n_players-1 (matches others)
    num_backgrounds = bg_noise_lf.shape[1]
    features = graph.features
    n_cols = n_coal * num_backgrounds  # column = coalition * B + draw

    base_sig_lf = np.concatenate([base_signals[f.name][:lf] for f in features])
    base_noise_lf = np.concatenate([base_noise[f.name][:lf] for f in features])

    # Membership (player-major): player i is present in coalition k iff its
    # permuted position is < k.
    present = _coalition_membership(inverse_permutation,
                                    n_players)  # (n_players, n_coal)

    # Fill each feature's buffers straight from its contiguous slice of players.
    signals, past_noise, fixed_past = {}, {}, {}
    for f_idx, f in enumerate(features):
        rows = slice(f_idx * lf, (f_idx + 1) * lf)
        innovations = _coalition_blocks(present[rows], base_noise_lf[rows],
                                        bg_noise_lf[rows], n_cols)
        signal = np.empty((seq_len, n_cols))
        signal[:lf] = innovations
        signal[lf:] = base_noise[f.name][lf:seq_len, None]
        signals[f.name] = signal
        if f.name == 'Hazard':
            continue  # propagates freely; carries no pinned value or weight
        past_noise[f.name] = innovations
        fixed_past[f.name] = _coalition_blocks(present[rows],
                                               base_sig_lf[rows],
                                               bg_sig_lf[rows], n_cols)

    # Pinned propagation: forces the pinned signals, returns per-column log-weight.
    perturbed_signals, logw = _apply_rules_obs(graph, signals, past_noise,
                                               fixed_past, seq_len, lf)
    hazard = hazard_calibration_func(perturbed_signals['Hazard'][lf:seq_len])
    sample_outputs = _outputs_from_hazard(hazard, hazard_bins)
    target = sample_outputs.get_target(output_type, hazard_bins)  # (n_cols,)

    # Self-normalised importance average over background draws -> v(S).
    logw = np.reshape(logw, (n_coal, num_backgrounds))
    target = np.reshape(target, (n_coal, num_backgrounds))
    w = np.exp(logw - logw.max(axis=1, keepdims=True))
    v = (w * target).sum(axis=1) / w.sum(axis=1)  # (n_coal,)

    return _contributions_from_values(v, permutation_order)


def _noise_observational(
    num_feats: int,
    permutation_order: np.ndarray,  # (n_players,)
    inverse_permutation: np.ndarray,  # (n_players,)
    graph: CausalGraph,
    base_noise: dict[str, np.ndarray],
    bg_noise_lf: np.ndarray,  # (n_players, num_backgrounds)
    lf: int,
    seq_len: int,
    hazard_calibration_func: callable,
    hazard_bins: np.ndarray | None,
    output_type: Literal['expected_time', 'binary'],
) -> np.ndarray:
    """Unweighted noise-swap observational baseline (kept for comparison).

    Same coalition/background grid as ``_observational``, but the innovations are
    simply propagated through the graph and the coalition value is the plain mean
    over the shared background draws — no signal pinning, no reweighting.
    """
    n_players = num_feats * lf  # players = (feature, frame) over [0, lf)
    n_coal = n_players
    num_backgrounds = bg_noise_lf.shape[1]
    features = graph.features
    n_cols = n_coal * num_backgrounds  # column = coalition * B + draw

    base_noise_lf = np.concatenate([base_noise[f.name][:lf] for f in features])

    present = _coalition_membership(inverse_permutation,
                                    n_players)  # (n_players, n_coal)

    perturbed_noises = {}
    for f_idx, f in enumerate(features):
        rows = slice(f_idx * lf, (f_idx + 1) * lf)
        noise = np.empty((seq_len, n_cols))
        noise[:lf] = _coalition_blocks(present[rows], base_noise_lf[rows],
                                       bg_noise_lf[rows], n_cols)
        noise[lf:] = base_noise[f.name][lf:seq_len, None]
        perturbed_noises[f.name] = noise

    # Built solely for this rollout, so propagate in place.
    perturbed_signals = _apply_rules(graph,
                                     perturbed_noises,
                                     seq_len,
                                     copy=False)
    hazard = hazard_calibration_func(perturbed_signals['Hazard'][lf:seq_len])

    sample_outputs = _outputs_from_hazard(hazard, hazard_bins)
    target = sample_outputs.get_target(output_type, hazard_bins)  # (n_cols,)

    # Plain (unweighted) mean over background draws -> v(S).
    target = np.reshape(target, (n_coal, num_backgrounds))
    v = target.mean(axis=1)  # (n_coal,)

    return _contributions_from_values(v, permutation_order)
