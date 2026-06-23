from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, fields, field

import numpy as np
from tqdm import tqdm

from .graph import CausalGraph


@dataclass
class outputs:
    hazard: np.ndarray
    pmf: np.ndarray
    cdf: np.ndarray


@dataclass
class importances:
    hazard: np.ndarray
    pmf: np.ndarray
    cdf: np.ndarray


@dataclass
class sampleWithImportances:
    base_signals: dict
    landmark_frame: int
    death_frame: int | None
    sample_importances: np.ndarray


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
        # if max_time_to_death is not None:
        #     lf = lf[death_frame - lf < max_time_to_death]
    if len(lf) == 0:
        return None
    lf = np.random.choice(lf)

    return lf, death_frame


def copy_signals(signals: dict) -> dict:
    return {k: v.copy() for k, v in signals.items()}


def generate_sample_with_feature_importance(
    graph: CausalGraph,
    hazard_calibration_func: callable,
    horizon: int,
    max_sequence_length: int,
    min_sequence_length: int,
    num_permutations: int = 100,
    hazard_bins: np.ndarray | None = None,
    output_type: Literal['expected_time', 'binary'] = 'expected_time'
) -> sampleWithImportances:

    # sample_importances = {f.name: [] for f in fields(outputs)}
    sample_importances = []

    full_seq_len = max_sequence_length + horizon
    base_noise = {
        f.name: f.generate_signal(full_seq_len)
        for f in graph.features
    }  # {feature_name: []}

    base_signals = _apply_rules(graph, copy_signals(base_noise),
                                max_sequence_length + horizon)
    landmark_sample = _landmark_sample(
        base_signals,
        hazard_calibration_func,
        max_sequence_length,
        min_sequence_length,
    )
    if landmark_sample is None:
        return None

    lf, death_frame = landmark_sample
    seq_len = lf + horizon

    num_feats = len(graph.features)
    total_num_feats = num_feats * lf
    total_feats = np.arange(total_num_feats)
    base_noise_arr = np.concatenate(
        [base_noise[f.name][:lf] for f in graph.features])

    # base_noise_arr = np.stack(base_noise.values(), axis=0)
    # perturbed_noises = np.stack()
    for _ in tqdm(range(num_permutations)):
        # Generate random perturbaiton
        permutation_order = np.random.permutation(total_feats)
        inverse_permutation = np.argsort(permutation_order)

        # Generate random backgorund noise sample
        background_sample = {
            f.name: f.generate_signal(lf)
            for f in graph.features
        }
        background_sample_arr = np.concatenate(list(
            background_sample.values()),
                                               axis=0)

        permuted_base = base_noise_arr[permutation_order]
        permuted_background = background_sample_arr[permutation_order]

        perturbed_noises = np.stack([permuted_base] * total_num_feats, axis=0)
        # background_mask = np.triu(np.ones(perturbed_noises.shape, dtype=bool))
        # perturbed_noises[background_mask] = permuted_background
        j = np.arange(total_num_feats)[None, :]  # (1, N) column (player index)
        k = np.arange(total_num_feats)[:, None]  # (N, 1) row (coalition size)
        perturbed_noises = np.where(j < k, permuted_base[None, :],
                                    permuted_background[None, :])

        # Reorder perturbed noises
        perturbed_noises = perturbed_noises[:, inverse_permutation]

        # Put pertubed noise back to dict formatting for running graph
        batch_noise = {}
        for f_idx, f in enumerate(graph.features):
            past = perturbed_noises[:, f_idx * lf:(f_idx + 1) * lf].T
            future = np.broadcast_to(base_noise[f.name][lf:seq_len, None],
                                     (horizon, total_num_feats))
            batch_noise[f.name] = np.concatenate([past, future],
                                                 axis=0)  # (seq_len, N)

        # perturbed_noises = np.reshape(
        #     perturbed_noises, (total_num_feats, num_feats, full_seq_len))
        # perturbed_noises = {
        #     f.name: perturbed_noises[:, i, :]
        #     for i, f in enumerate(graph.features)
        # }

        perturbed_signals = _apply_rules(graph, batch_noise, seq_len)
        hazard = hazard_calibration_func(
            perturbed_signals['Hazard'][lf:seq_len])

        # Get sample outputs
        sample_outputs = _outputs_from_hazard(hazard, hazard_bins)
        if output_type == 'expected_time':
            survival = 1.0 - np.cumsum(sample_outputs.pmf,
                                       axis=0)  # (n_bins, B)
            if hazard_bins is not None:
                bin_widths = np.diff(hazard_bins)
                sample_outputs = (survival * bin_widths[:, None]).sum(
                    axis=0)  # (B,)
            else:
                sample_outputs = survival.sum(axis=0)  # (B,)
        elif output_type == 'binary':
            sample_outputs = sample_outputs.cdf[-1]  # (B,)

        # Get shapley contributions
        marginals = np.diff(sample_outputs)
        contributions = np.zeros_like(sample_outputs)
        contributions[permutation_order[:-1]] = marginals
        sample_importances.append(contributions)

    sample_importances = np.stack(sample_importances, axis=0)
    sample_importances = np.mean(sample_importances, axis=0)

    sample_importances = sample_importances.reshape(num_feats, lf)

    return sampleWithImportances(base_signals, lf, death_frame,
                                 sample_importances)


def generate_sample_with_feature_importance_batched(
    graph: CausalGraph,
    hazard_calibration_func: callable,
    horizon: int,
    max_sequence_length: int,
    min_sequence_length: int,
    num_permutations: int = 100,
    hazard_bins: np.ndarray | None = None,
    output_type: Literal['expected_time', 'binary'] = 'expected_time',
    permutations_per_batch: int = 50,
) -> sampleWithImportances | None:
    """Same as generate_sample_with_feature_importance but runs permutations_per_batch
    permutations simultaneously by stacking their coalition noise arrays into a single
    (seq_len, M*N) batch and calling _apply_rules once per batch instead of once per
    permutation."""

    sample_importances = []

    full_seq_len = max_sequence_length + horizon
    base_noise = {
        f.name: f.generate_signal(full_seq_len)
        for f in graph.features
    }

    base_signals = _apply_rules(graph, copy_signals(base_noise),
                                max_sequence_length + horizon)
    landmark_sample = _landmark_sample(
        base_signals,
        hazard_calibration_func,
        max_sequence_length,
        min_sequence_length,
    )
    if landmark_sample is None:
        return None

    lf, death_frame = landmark_sample
    seq_len = lf + horizon

    num_feats = len(graph.features)
    total_num_feats = num_feats * lf
    N = total_num_feats
    total_feats = np.arange(N)
    base_noise_arr = np.concatenate(
        [base_noise[f.name][:lf] for f in graph.features])

    for batch_start in tqdm(range(0, num_permutations,
                                  permutations_per_batch)):
        M = min(permutations_per_batch, num_permutations - batch_start)

        # Generate M permutations and background samples
        permutation_orders = [
            np.random.permutation(total_feats) for _ in range(M)
        ]
        inverse_permutations = [np.argsort(p) for p in permutation_orders]
        background_sample_arrs = [
            np.concatenate([f.generate_signal(lf) for f in graph.features])
            for _ in range(M)
        ]

        # Vectorise over all M permutations simultaneously
        permutation_orders_arr = np.stack(permutation_orders)  # (M, N)
        inv_perms = np.stack(inverse_permutations)  # (M, N)
        background_arr = np.stack(background_sample_arrs)  # (M, N)

        permuted_bases = base_noise_arr[permutation_orders_arr]  # (M, N)
        permuted_backgrounds = background_arr[  # (M, N)
            np.arange(M)[:, None], permutation_orders_arr]

        # Build (M, N, N): row k uses base for players j < k, background otherwise
        j = np.arange(N)[None, None, :]
        k = np.arange(N)[None, :, None]
        perturbed_noises_all = np.where(
            j < k, permuted_bases[:, None, :],
            permuted_backgrounds[:, None, :])  # (M, N, N)

        # Reorder columns to player-order for all M permutations at once
        perturbed_noises_all = perturbed_noises_all[
            np.arange(M)[:, None, None],
            np.arange(N)[None, :, None], inv_perms[:, None, :]]  # (M, N, N)

        batch_noise = {}
        for f_idx, f in enumerate(graph.features):
            past_all = perturbed_noises_all[:, :, f_idx * lf:(f_idx + 1) *
                                            lf]  # (M, N, lf)
            past_stacked = past_all.transpose(2, 0,
                                              1).reshape(lf,
                                                         M * N)  # (lf, M*N)
            future_stacked = np.broadcast_to(
                base_noise[f.name][lf:seq_len, None], (horizon, M * N))
            batch_noise[f.name] = np.concatenate(
                [past_stacked, future_stacked], axis=0)

        # Single simulation for all M*N coalitions
        perturbed_signals = _apply_rules(graph, batch_noise, seq_len)
        hazard = hazard_calibration_func(
            perturbed_signals['Hazard'][lf:seq_len])  # (horizon, M*N)

        sample_outputs = _outputs_from_hazard(hazard, hazard_bins)
        if output_type == 'expected_time':
            survival = 1.0 - np.cumsum(sample_outputs.pmf, axis=0)
            if hazard_bins is not None:
                bin_widths = np.diff(hazard_bins)
                sample_outputs = (survival * bin_widths[:, None]).sum(axis=0)
            else:
                sample_outputs = survival.sum(axis=0)  # (M*N,)
        elif output_type == 'binary':
            sample_outputs = sample_outputs.cdf[-1]  # (M*N,)

        # Reshape to (M, N), extract per-permutation Shapley contributions
        sample_outputs_batch = sample_outputs.reshape(M, N)
        for m in range(M):
            marginals = np.diff(sample_outputs_batch[m])
            contributions = np.zeros(N)
            contributions[permutation_orders[m][:-1]] = marginals
            sample_importances.append(contributions)

    sample_importances = np.stack(sample_importances, axis=0)
    sample_importances = np.mean(sample_importances, axis=0)
    sample_importances = sample_importances.reshape(num_feats, lf)

    return sampleWithImportances(base_signals, lf, death_frame,
                                 sample_importances)


# def generate_sample_with_feature_importance(
#         graph: CausalGraph,
#         hazard_calibration_func: callable,
#         horizon: int,
#         max_sequence_length: int,
#         min_sequence_length: int,
#         samples_per_node: int = 10,
#         hazard_bins: np.ndarray
#     | None = None) -> sampleWithImportances:
#     base_noise = {
#         f.name: f.generate_signal(max_sequence_length + horizon)
#         for f in graph.features
#     }  # {feature_name: []}

#     base_signals = _apply_rules(graph, copy_signals(base_noise),
#                                 max_sequence_length + horizon)
#     landmark_sample = _landmark_sample(
#         base_signals,
#         hazard_calibration_func,
#         max_sequence_length,
#         min_sequence_length,
#     )
#     if landmark_sample is None:
#         return None

#     lf, death_frame = landmark_sample
#     seq_len = lf + horizon
#     perturbation_times = np.repeat(np.arange(lf), samples_per_node)

#     # sample_importances = importances([])
#     sample_importances = {f.name: [] for f in fields(outputs)}

#     for feature in tqdm(graph.features, desc='Getting importances'):
#         noise_perturbations = feature.generate_signal(samples_per_node * lf)
#         noise = {
#             f.name:
#             np.broadcast_to(base_noise[f.name][:seq_len, None],
#                             (seq_len, samples_per_node * lf)).copy()
#             for f in graph.features
#         }
#         noise[feature.name][perturbation_times,
#                             np.arange(lf *
#                                       samples_per_node)] = noise_perturbations
#         signals = _apply_rules(graph, noise, seq_len)
#         hazard = hazard_calibration_func(signals['Hazard'][lf:seq_len])
#         sample_outputs = _outputs_from_hazard(hazard, hazard_bins)

#         # sample_importances = importances()

#         for f in fields(outputs):
#             sample_output = getattr(sample_outputs, f.name)
#             sample_output = np.reshape(sample_output,
#                                        (-1, lf, samples_per_node))
#             variances = np.var(sample_output, axis=2)
#             sample_importances[f.name].append(variances)

#     sample_importances = {
#         k: np.stack(v, axis=0)
#         for k, v in sample_importances.items()
#     }
#     sample_importances = importances(**sample_importances)

#     return sampleWithImportances(base_signals, lf, death_frame,
#                                  sample_importances)
