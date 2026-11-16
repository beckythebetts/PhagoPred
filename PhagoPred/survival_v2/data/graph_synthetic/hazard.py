"""Hazard functions: mapping the graph's Hazard signal to per-frame event
rates, calibrating that mapping, and turning rates into event-time
distributions."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit

from .graph import CausalGraph


def _offset_sigmoid(x: np.ndarray,
                    delta: float = 0.0,
                    scale: float = 1.0) -> np.ndarray:
    """h(x) = 1 / (1 + exp(delta - scale * x)). ``scale`` sets how spread out
    the rates are, and so how predictable deaths are; ``delta`` sets how many
    cells die."""
    return expit(scale * x - delta)


_HAZARD_FUNCS = {
    'offset_sigmoid': _offset_sigmoid,
}


@dataclass
class HazardFunc:
    """Maps the graph's Hazard signal to per-frame event rates: which kind,
    and its parameters (fitted by calibrate_hazard). Plain values only, so it
    saves with the scenario and loads with HazardFunc(**d)."""
    kind: str = 'offset_sigmoid'
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _HAZARD_FUNCS:
            raise ValueError(f'unknown hazard func {self.kind!r}; '
                             f'known: {list(_HAZARD_FUNCS)}')
        self(np.zeros(1))  # fail now on bad params

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _HAZARD_FUNCS[self.kind](x, **self.params)

    @classmethod
    def offset_sigmoid(cls,
                       delta: float = 0.0,
                       scale: float = 1.0) -> HazardFunc:
        return cls('offset_sigmoid', {'delta': delta, 'scale': scale})


def pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """Per-frame event rates (time on axis 0) -> probability of the event
    happening at each frame."""
    hazards = np.clip(hazards, 0.0, 1.0)
    sf = np.cumprod(1.0 - hazards, axis=0)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def bin_hazards(hazards: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Per-frame rates (time on axis 0) -> one rate per bin [bins[i],
    bins[i+1]): the probability of the event within the bin, given survival
    to its start."""
    bins = np.asarray(bins, dtype=int)
    return np.stack([
        1.0 - np.prod(1.0 - hazards[bins[i]:bins[i + 1]], axis=0)
        for i in range(len(bins) - 1)
    ])


def _expected_death_fraction(
    pre_hazard: np.ndarray,
    window_hazard: np.ndarray,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    offset: float,
    scale: float = 1.0,
) -> float:
    """Expected fraction of cells dying within [start, end] of the window, for
    a given hazard offset, among cells that survive the death-active burn-in.

    ``pre_hazard`` (death_burn_in, N) and ``window_hazard`` (T, N) are raw
    Hazard signals from one fixed pool. Survival through the burn-in is
    applied as a weight rather than by rejection sampling, so the result is a
    smooth, noise-free function of ``offset``.
    """
    hazard_func = HazardFunc.offset_sigmoid(offset, scale)
    weights = np.prod(1.0 - np.clip(hazard_func(pre_hazard), 0.0, 1.0), axis=0)
    rates = np.clip(hazard_func(window_hazard), 0.0, 1.0)
    cif = np.cumsum(pmf_from_hazards(rates), axis=0)
    cells = np.arange(cif.shape[1])
    end = np.clip(end_frames.astype(int), 0, cif.shape[0] - 1)
    start = start_frames.astype(int)
    before_start = np.where(start > 0, cif[np.maximum(start - 1, 0), cells],
                            0.0)
    p_death = cif[end, cells] - before_start
    if weights.sum() <= 0:
        return 0.0
    return float(np.sum(weights * p_death) / weights.sum())


def sample_death_frames(hazard_rates: np.ndarray) -> np.ndarray:
    """Draw one event frame per column from per-frame rates (time on axis 0):
    the first frame where the CIF reaches a uniform draw, NaN if it never
    does (censored by the end of the window)."""
    cif = np.cumsum(pmf_from_hazards(hazard_rates), axis=0)  # (T, N)
    u = np.random.rand(*cif.shape[1:])  # (N,)
    frames = np.argmax(cif >= u, axis=0).astype(float)
    frames[u > cif.max(axis=0)] = np.nan
    return frames


def calibrate_hazard(
    graph: CausalGraph,
    num_frames: int,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    target_death_fraction: float,
    hazard_spread: float | None = 3.0,
    spread_start: int = 0,
    n_calib: int = 1000,
    max_iter: int = 50,
    tol: float = 0.02,
    step_init: float = 10.0,
    step_growth: float = 2.0,
    step_max: float = 1e6,
) -> HazardFunc:
    """Find a scalar sigmoid offset so that ~target_death_fraction of cells die
    within their window, via geometric bracketing then bisection.

    ``observed(offset)`` is monotone *decreasing* (larger offset -> smaller
    hazard -> fewer deaths).  The previous analytical log-odds update assumed a
    constant per-frame hazard, which badly underestimates the required offset
    when the hazard signal is spiky (the true response is nearly flat), so it
    crawled.  Instead we bracket the target by growing the search step
    geometrically, then bisect the bracket.  This is robust to a flat response
    and to the Monte-Carlo noise in ``observed``, and if the target is
    unreachable (a floor above/below it) the bracketing phase says so rather
    than iterating forever.  The offset giving the closest observed death
    fraction is tracked and returned, so a sensible result comes back even if
    ``tol`` is never met exactly.

    One pool of ``n_calib`` cells is simulated up front (including the
    graph's burn-in) and reused for every offset; cells dying during the
    death-active burn-in are accounted for by survival weights (see
    ``_expected_death_fraction``), matching the rejection used when the
    datasets are generated.

    Before the offset search, ``scale`` is set so the scaled Hazard signal
    has a standard deviation of ``hazard_spread`` over the pool's window,
    making scenarios comparably predictable whatever the scale of their
    Hazard rule's output (None keeps scale 1). The spread is measured on the
    raw signal, so it does not depend on the offset, over window frames from
    ``spread_start`` on (skips the start-up ramp when the hazard only starts
    at frame 0)."""
    iters = 0
    best_offset = 0.0
    best_err = float('inf')

    pool = graph.simulate(n_calib,
                          num_frames,
                          keep_context=graph.death_burn_in,
                          post_noise=False)['Hazard']
    pre_hazard = pool[:graph.death_burn_in]
    window_hazard = pool[graph.death_burn_in:]
    start_frames = np.resize(np.asarray(start_frames), n_calib)
    end_frames = np.resize(np.asarray(end_frames), n_calib)

    scale = 1.0
    raw_spread = float(window_hazard[spread_start:].std())
    if hazard_spread is not None and raw_spread > 0:
        scale = hazard_spread / raw_spread
    print(f"  Hazard spread {raw_spread:.4g} -> scale {scale:.4g} "
          f"(target spread {hazard_spread})")

    def evaluate(offset: float, phase: str) -> float:
        nonlocal iters, best_offset, best_err
        obs = _expected_death_fraction(pre_hazard, window_hazard, start_frames,
                                       end_frames, offset, scale)
        err = abs(obs - target_death_fraction)
        if err < best_err:
            best_err = err
            best_offset = offset
        iters += 1
        print(f"  Calibration iter {iters} [{phase}]: offset={offset:.4f}, "
              f"observed={obs:.3f}, target={target_death_fraction:.3f}, "
              f"best_err={best_err:.3f}")
        return obs

    # --- Phase 1: bracket the target -----------------------------------
    # Seek lo with observed(lo) >= target and hi with observed(hi) <= target.
    offset = 0.0
    obs = evaluate(offset, "bracket")
    if best_err <= tol:
        return HazardFunc.offset_sigmoid(delta=best_offset, scale=scale)

    step = step_init
    lo = hi = None
    if obs > target_death_fraction:
        # Too many deaths -> raise offset until observed drops to/below target.
        lo, lo_obs = offset, obs
        while iters < max_iter:
            offset += step
            obs = evaluate(offset, "bracket")
            if best_err <= tol:
                return HazardFunc.offset_sigmoid(delta=best_offset,
                                                 scale=scale)
            if obs <= target_death_fraction:
                hi = offset
                break
            lo, lo_obs = offset, obs
            step = min(step * step_growth, step_max)
        if hi is None:
            print(f"  Calibration warning: target {target_death_fraction:.3f} "
                  f"unreachable; observed floored at {lo_obs:.3f} by "
                  f"offset {lo:.4f}. Returning best offset.")
            return HazardFunc.offset_sigmoid(delta=best_offset, scale=scale)
    else:
        # Too few deaths -> lower offset until observed rises to/above target.
        hi, hi_obs = offset, obs
        while iters < max_iter:
            offset -= step
            obs = evaluate(offset, "bracket")
            if best_err <= tol:
                return HazardFunc.offset_sigmoid(delta=best_offset,
                                                 scale=scale)
            if obs >= target_death_fraction:
                lo = offset
                break
            hi, hi_obs = offset, obs
            step = min(step * step_growth, step_max)
        if lo is None:
            print(f"  Calibration warning: target {target_death_fraction:.3f} "
                  f"unreachable; observed capped at {hi_obs:.3f} by "
                  f"offset {hi:.4f}. Returning best offset.")
            return HazardFunc.offset_sigmoid(delta=best_offset, scale=scale)

    # --- Phase 2: bisection --------------------------------------------
    # Invariant: lo < hi and observed(lo) >= target >= observed(hi).
    while iters < max_iter:
        mid = 0.5 * (lo + hi)
        obs = evaluate(mid, "bisect")
        if best_err <= tol:
            break
        if obs > target_death_fraction:
            lo = mid
        else:
            hi = mid

    return HazardFunc.offset_sigmoid(delta=best_offset, scale=scale)
