from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .graph import CausalGraph


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    sf = np.cumprod(1.0 - hazards, axis=0)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def _apply_rules(graph: CausalGraph, noise: dict[str, np.ndarray],
                 time_steps: int) -> dict[str, np.ndarray]:
    # == OPTIONALLY ADD POST_NOISE===
    signals = noise.copy()
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
    return signals


def _outputs_from_hazard(hazard: np.ndarray,
                         hazard_bins: np.ndarray | None = None):
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
    return {'hazard': hazard, 'cdf': cdf, 'pmf': pmf}


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
        # if max_time_to_death is not None:
        #     lf = lf[death_frame - lf < max_time_to_death]
    if len(lf) == 0:
        return None
    lf = np.random.choice(lf)


def copy_signals(signals: dict) -> dict:
    return {k: v.copy() for k, v in signals.items()}


def generate_sample_with_feature_importance(graph: CausalGraph,
                                            hazard_calibration_func: callable,
                                            horizon: int,
                                            max_sequence_length: int,
                                            min_sequence_length: int,
                                            samples_per_node: int = 10,
                                            hazard_bins: np.ndarray
                                            | None = None):
    base_noise = {
        f.name: f.generate_signal(max_sequence_length)
        for f in graph.features
    }

    base_signals = _apply_rules(graph, base_noise, max_sequence_length)
    lf = _landmark_sample(
        base_signals,
        hazard_calibration_func,
        max_sequence_length,
        min_sequence_length,
    )
    if lf is None:
        return None

    importance = 
