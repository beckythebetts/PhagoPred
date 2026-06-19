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


def _apply_rules_batch(graph: CausalGraph, signals: dict[str, np.ndarray],
                       time_steps: int) -> dict[str, np.ndarray]:
    """Propagate the graph rules over a *batched* signals dict in place.

    ``signals[name]`` has shape (time_steps, B); every rule's numpy ops broadcast
    over the trailing batch axis, so B independent trajectories evolve at once.
    Post-noise is intentionally omitted to match ``estimate_variances`` (the
    hazard is read straight off the rule-propagated signal).
    """
    for t in range(time_steps):
        for rule in graph.rules:
            rule.apply_step(signals, t)
    return signals


def _outcomes_from_horizon_hazard(
        horizon_hazard: np.ndarray,
        hazard_bins: np.ndarray | None) -> dict[str, np.ndarray]:
    """Turn a (horizon, B) hazard array into hazard / cdf / pmf outcomes.

    Mirrors ``estimate_variances``: when ``hazard_bins`` is given the hazard and
    pmf are collapsed onto the bins (cdf is always kept at horizon resolution).
    """
    cdf = np.cumsum(_pmf_from_hazards(horizon_hazard), axis=0)  # (horizon, B)

    if hazard_bins is not None:
        bins = np.asarray(hazard_bins, dtype=int)
        hazard = np.stack([
            1.0 - np.prod(1.0 - horizon_hazard[bins[i]:bins[i + 1]], axis=0)
            for i in range(len(bins) - 1)
        ])  # (n_bins, B)
    else:
        hazard = horizon_hazard  # (horizon, B)

    pmf = _pmf_from_hazards(hazard)
    return {'hazard': hazard, 'cdf': cdf, 'pmf': pmf}


def generate_sample_with_ground_truth(graph: CausalGraph,
                                      hazard_calibration_func: callable,
                                      horizon: int,
                                      max_sequence_length: int,
                                      min_sequence_length: int,
                                      samples_per_node: int = 10,
                                      hazard_bins: np.ndarray | None = None):
    """Ground-truth temporal-feature importance for one synthetic cell.
    """
    feature_names = [f.name for f in graph.features]
    hazard_name = 'Hazard'

    base_noise = {
        f.name: f.generate_signal(max_sequence_length)
        for f in graph.features
    }

    base_signals = _apply_rules_batch(graph, {
        k: v.copy()
        for k, v in base_noise.items()
    }, max_sequence_length)

    base_hazard = hazard_calibration_func(base_signals[hazard_name])
    base_cif = np.cumsum(_pmf_from_hazards(base_hazard))
    u = np.random.rand()
    death_frame = float(np.argmax(base_cif >= u)) if u <= base_cif.max() \
        else None

    lf = np.arange(max_sequence_length)
    lf = lf[lf > min_sequence_length]
    lf = lf[lf + horizon <= max_sequence_length]
    if death_frame is not None:
        lf = lf[lf <= death_frame]
    if len(lf) == 0:
        return None
    lf = int(np.random.choice(lf))
    seq_len = lf + horizon

    spn = samples_per_node
    B = lf * spn
    perturb_t = np.repeat(np.arange(lf), spn)  # (B,)
    cols = np.arange(B)

    importance: dict[str, np.ndarray] | None = None
    for f_idx, feature in tqdm(enumerate(graph.features),
                               desc='Getting importances'):
        # Broadcast the fixed base noise across all columns, then overwrite a
        # single (time_step, column) cell of THIS feature with a fresh draw.
        signals = {
            name:
            np.broadcast_to(base_noise[name][:seq_len, None],
                            (seq_len, B)).copy()
            for name in feature_names
        }
        signals[feature.name][perturb_t, cols] = feature.generate_signal(B)

        _apply_rules_batch(graph, signals, seq_len)
        horizon_hazard = hazard_calibration_func(
            signals[hazard_name][lf:seq_len])  # (horizon, B)
        outcomes = _outcomes_from_horizon_hazard(horizon_hazard, hazard_bins)

        if importance is None:
            importance = {
                q: np.zeros((len(graph.features), lf, arr.shape[0]))
                for q, arr in outcomes.items()
            }
        for q, arr in outcomes.items():
            k = arr.shape[0]
            # (k, B) -> (k, lf, spn); variance across the spn resamples.
            var = arr.reshape(k, lf, spn).var(axis=2)  # (k, lf)
            importance[q][f_idx] = var.T  # (lf, k)

    primary = 'pmf' if hazard_bins is not None else 'cdf'
    return {
        'landmark_frame': lf,
        'death_frame': death_frame,
        'horizon': horizon,
        'feature_names': feature_names,
        'hazard_bins':
        None if hazard_bins is None else np.asarray(hazard_bins),
        'primary_quantity': primary,
        # importance[q]: (n_features, lf, k) variance per (feature, time_step).
        'importance': importance,
        # convenience scalar per (feature, time_step): mean over the k outcome
        # components, comparable to SHAP temporal_feature_importance.
        'importance_scalar': {
            q: arr.mean(axis=2)
            for q, arr in importance.items()
        },
    }
