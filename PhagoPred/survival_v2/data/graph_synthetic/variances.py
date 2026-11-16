from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .scenarios import ScenarioCfg
from .hazard import bin_hazards, pmf_from_hazards, sample_death_frames
from .dynamics import branch


@dataclass
class Variances:
    event_probability: float
    hazard: np.ndarray | None
    pmf: np.ndarray | None


def estimate_unobserved_variance(
    scenario: ScenarioCfg,
    horizon: int,
    min_sequence_length: int,
    landmark_rule: callable,
    hazard_bins: np.ndarray | None = None,
    trunk_sample_size: int = 500,
    branch_sample_size=1000,
) -> Variances:
    graph, hazard_func = scenario.graph, scenario.hazard_func
    num_frames = scenario.num_frames
    missingness = scenario.missingness

    _event_probs = []
    _hazards = [] if hazard_bins is not None else None
    _pmfs = [] if hazard_bins is not None else None
    _weights = []

    ctx = graph.max_lag

    trunks = graph.simulate(trunk_sample_size,
                            num_frames,
                            hazard_func,
                            keep_context=ctx,
                            post_noise=False)  # (ctx + T, n)
    deaths = sample_death_frames(hazard_func(trunks['Hazard'][ctx:]))  # (n,)
    starts, ends = missingness.sample_observed_ranges(num_frames,
                                                      trunk_sample_size)

    deaths[(deaths < starts) | (deaths > ends)] = np.nan

    for i in tqdm(range(trunk_sample_size),
                  desc='Estimating unobserved variances'):

        landmarks = landmark_rule(starts[i], ends[i], deaths[i],
                                  min_sequence_length, horizon)
        if len(landmarks) == 0:
            continue

        _weights.append(len(landmarks))
        lf = np.random.choice(landmarks)

        trunk_signals = {
            name: sig[:lf + ctx, i]
            for name, sig in trunks.items()
        }
        branched_signals = branch(
            graph,
            trunk_signals,
            horizon + 1,
            branch_sample_size,
            frame_0=ctx,
        )
        hazards = hazard_func(branched_signals['Hazard'])
        hazards[max(num_frames - lf, 0):] = 0.0
        event_probs = 1.0 - np.prod(1.0 - hazards[1:], axis=0)
        _event_probs.append(np.var(event_probs, ddof=1))
        if hazard_bins is not None:
            hazards = bin_hazards(hazards, hazard_bins)
            _hazards.append(np.var(hazards, axis=1, ddof=1))
            _pmfs.append(np.var(pmf_from_hazards(hazards), axis=1, ddof=1))

    weights = np.asarray(_weights, dtype=float)
    weights /= weights.sum()
    return Variances(
        event_probability=float(np.dot(weights, _event_probs)),
        hazard=None if hazard_bins is None else weights @ np.asarray(_hazards),
        pmf=None if hazard_bins is None else weights @ (np.asarray(_pmfs)),
    )
