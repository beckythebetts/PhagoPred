from __future__ import annotations
from pathlib import Path
import math
from typing import Literal

import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .graph import CausalGraph, Feature
from .rules import Rule


class OffsetSigmoid:
    """Sigmoid hazard transform: h(x) = 1 / (1 + exp(delta - x)).

    delta is fitted by calibrate_hazard so that the observed death fraction
    matches the target.  Instances can be saved to / loaded from HDF5
    metadata via to_metadata / from_metadata.
    """
    name = 'offset_sigmoid'

    def __init__(self, delta: float = 0.0):
        self.delta = delta

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(self.delta - x))

    def to_metadata(self) -> dict:
        return {'hazard_func': self.name, 'delta': self.delta}

    @classmethod
    def from_metadata(cls, attrs) -> OffsetSigmoid:
        return cls(delta=float(attrs['delta']))


_HAZARD_FUNCS: dict[str, type] = {
    OffsetSigmoid.name: OffsetSigmoid,
}


def load_hazard_func(attrs) -> OffsetSigmoid:
    """Reconstruct a hazard calibration object from HDF5 metadata attributes."""
    name = str(attrs['hazard_func'])
    if name not in _HAZARD_FUNCS:
        raise ValueError(
            f"Unknown hazard func '{name}'. Known: {list(_HAZARD_FUNCS)}")
    return _HAZARD_FUNCS[name].from_metadata(attrs)


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    sf = np.cumprod(1.0 - hazards, axis=0)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def _observe_death_fraction(
    graph: CausalGraph,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    offset: float,
    n_samples: int,
) -> float:
    """Estimate the death fraction for a given hazard offset over n_samples cells."""
    hazard_func = OffsetSigmoid(offset)
    deaths = 0
    for c in range(n_samples):
        signals = graph.sample_graph()
        hazard_rates = hazard_func(signals['Hazard'].copy())
        pmf = _pmf_from_hazards(hazard_rates)
        cif = np.cumsum(pmf)
        start, end = int(start_frames[c]), int(end_frames[c])
        u = np.random.rand()
        if u <= cif.max():
            df = float(np.argmax(cif >= u))
            if start <= df <= end:
                deaths += 1
    return deaths / n_samples


def calibrate_hazard(
    graph: CausalGraph,
    num_frames: int,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    target_death_fraction: float,
    n_calib: int = 200,
    max_iter: int = 50,
    tol: float = 0.02,
) -> OffsetSigmoid:
    """Iteratively find a scalar offset when applying sigmoid to hazard rates
    so that approximately target_death_fraction of cells die within their window."""
    offset = 0.0
    step = 10.0
    for i in range(max_iter):
        observed = _observe_death_fraction(
            graph,
            start_frames,
            end_frames,
            offset=offset,
            n_samples=n_calib,
        )
        print(f"  Calibration iter {i + 1}: offset={offset:.4f}, "
              f"observed={observed:.3f}, target={target_death_fraction:.3f}")
        if abs(observed - target_death_fraction) <= tol:
            break
        if observed < 1e-6:
            offset -= step
        elif observed >= 1 - 1e-6:
            offset += step
        else:
            offset = offset + math.log(
                1 / (1 - (1 - target_death_fraction)**(1 / num_frames)) -
                1) - math.log(1 / (1 - (1 - observed)**(1 / num_frames)) - 1)

    return OffsetSigmoid(delta=offset)


def _generate_cells(
    graph: CausalGraph,
    num_cells: int,
    num_frames: int,
    late_entry_prob: float,
    late_entry_range: tuple,
    early_exit_prob: float,
    early_exit_range: tuple,
    feature_mask_prob: float,
    hazard_calibration_func: OffsetSigmoid,
    non_hazard_names: list,
) -> tuple:
    """Generate cell trajectories using a pre-calibrated hazard offset."""
    start_frames = np.zeros(num_cells, dtype=int)
    if late_entry_prob > 0.0:
        late_mask = np.random.rand(num_cells) < late_entry_prob
        start_frames[late_mask] = np.random.randint(late_entry_range[0],
                                                    late_entry_range[1],
                                                    size=int(late_mask.sum()))
    else:
        start_frames[:] = np.random.randint(0, 10, size=num_cells)

    end_frames = np.full(num_cells, num_frames - 1, dtype=int)
    if early_exit_prob > 0.0:
        early_mask = np.random.rand(num_cells) < early_exit_prob
        end_frames[early_mask] = np.random.randint(early_exit_range[0],
                                                   early_exit_range[1],
                                                   size=int(early_mask.sum()))
    else:
        end_frames[:] = np.random.randint(num_frames - 10,
                                          num_frames,
                                          size=num_cells)

    all_features = {
        name: np.full((num_frames, num_cells), np.nan, dtype=np.float32)
        for name in non_hazard_names
    }
    all_hazard_signals = np.full((num_frames, num_cells),
                                 np.nan,
                                 dtype=np.float32)
    all_hazard_rates = np.empty((num_frames, num_cells), dtype=np.float32)
    all_pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    all_cifs = np.empty((num_frames, num_cells), dtype=np.float32)
    all_deaths = np.full(num_cells, np.nan, dtype=np.float32)

    for c in tqdm(range(num_cells), desc='Generating cells'):
        signals = graph.sample_graph()
        start, end = int(start_frames[c]), int(end_frames[c])

        hazard_signal = signals['Hazard'].copy()
        hazard_rates = hazard_calibration_func(hazard_signal).astype(
            np.float32)

        pmf = _pmf_from_hazards(hazard_rates).astype(np.float32)
        cif = np.cumsum(pmf).astype(np.float32)

        all_hazard_rates[:, c] = hazard_rates
        all_pmfs[:, c] = pmf
        all_cifs[:, c] = cif

        u = np.random.rand()
        if u > cif.max():
            death_frame = np.nan
        else:
            death_frame = float(np.argmax(cif >= u))
            if death_frame < start or death_frame > end:
                death_frame = np.nan
        all_deaths[c] = death_frame

        for name in non_hazard_names:
            sig = signals[name].astype(np.float32)
            sig[:start] = np.nan
            sig[end + 1:] = np.nan
            if feature_mask_prob > 0.0:
                mask = np.random.rand(num_frames) < feature_mask_prob
                sig[mask] = np.nan
            all_features[name][:, c] = sig

        hz_sig = hazard_signal.astype(np.float32)
        hz_sig[:start] = np.nan
        hz_sig[end + 1:] = np.nan
        all_hazard_signals[:, c] = hz_sig

    return all_features, all_hazard_signals, all_hazard_rates, all_pmfs, all_cifs, all_deaths


def _save_dataset(
    filename: Path,
    all_features: dict,
    all_hazard_signals: np.ndarray,
    all_hazard_rates: np.ndarray,
    all_pmfs: np.ndarray,
    all_cifs: np.ndarray,
    all_deaths: np.ndarray,
    feature_names: list,
    non_hazard_names: list,
    num_frames: int,
    late_entry_prob: float,
    feature_mask_prob: float,
    hazard_calibration_func: OffsetSigmoid,
) -> None:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    num_cells = all_deaths.shape[0]

    with h5py.File(filename, 'w') as f:
        grp = f.create_group('Cells/Phase')

        for name in non_hazard_names:
            grp.create_dataset(name, data=all_features[name], dtype=np.float32)

        grp.create_dataset('Hazard', data=all_hazard_signals, dtype=np.float32)
        grp.create_dataset('HazardRates',
                           data=all_hazard_rates,
                           dtype=np.float32)
        grp.create_dataset('PMFs', data=all_pmfs, dtype=np.float32)
        grp.create_dataset('CIFs', data=all_cifs, dtype=np.float32)
        grp.create_dataset('CellDeath',
                           data=all_deaths[np.newaxis, :],
                           dtype=np.float32)

        cells = f['Cells']
        cells.attrs['num_cells'] = num_cells
        cells.attrs['num_frames'] = num_frames
        cells.attrs['late_entry_prob'] = late_entry_prob
        cells.attrs['feature_mask_prob'] = feature_mask_prob
        cells.attrs['feature_names'] = feature_names
        for key, val in hazard_calibration_func.to_metadata().items():
            cells['Phase']['HazardRates'].attrs[key] = val


def generate_dataset(
    train_filename: Path,
    val_filename: Path,
    graph: CausalGraph,
    train_num_cells: int = 1000,
    val_num_cells: int = 200,
    cal_filename: Path | None = None,
    cal_num_cells: int = 200,
    num_frames: int = 500,
    late_entry_prob: float = 0.0,
    late_entry_range: tuple = (0, 100),
    early_exit_prob: float = 0.0,
    early_exit_range: float = (0, 100),
    feature_mask_prob: float = 0.0,
    hazard_calibration_func: OffsetSigmoid | None = None,
    seed: int = None,
):
    """Generate train and validation synthetic survival datasets from a CausalGraph.

    Calibrates the hazard offset once (from target_death_fraction) then
    uses it for both splits, ensuring they share the same hazard scaling.

    Args:
        train_filename: Output .h5 path for the training set.
        val_filename: Output .h5 path for the validation set.
        graph: CausalGraph whose features must include one named 'Hazard'.
        train_num_cells: Number of cells in the training set.
        val_num_cells: Number of cells in the validation set.
        num_frames: Number of time frames per trajectory.
        late_entry_prob: Fraction of cells with a late observation start.
        late_entry_range: (min, max) frame range for late entry start times.
        feature_mask_prob: Per-frame probability of masking feature values with NaN.
        hazard_calibration_func: Pre-calibrated hazard func.  If None, an
            OffsetSigmoid with delta=0 is used (no calibration).
        seed: Random seed for reproducibility. Val set uses seed+1.
    Returns:
        The hazard calibration object used (useful if it was passed in pre-calibrated).
    """
    if seed is not None:
        np.random.seed(seed)

    feature_names = [f.name for f in graph.features]
    assert 'Hazard' in feature_names, "Features must include a node named 'Hazard'."
    non_hazard_names = [n for n in feature_names if n != 'Hazard']

    if hazard_calibration_func is None:
        hazard_calibration_func = OffsetSigmoid()

    print(
        f"Generating training set ({train_num_cells} cells) -> {train_filename}"
    )
    train_data = _generate_cells(
        graph,
        train_num_cells,
        num_frames,
        late_entry_prob,
        late_entry_range,
        early_exit_prob,
        early_exit_range,
        feature_mask_prob,
        hazard_calibration_func,
        non_hazard_names,
    )
    _save_dataset(train_filename, *train_data, feature_names, non_hazard_names,
                  num_frames, late_entry_prob, feature_mask_prob,
                  hazard_calibration_func)

    if seed is not None:
        np.random.seed(seed + 1)
    print(
        f"Generating validation set ({val_num_cells} cells) -> {val_filename}")
    val_data = _generate_cells(
        graph,
        val_num_cells,
        num_frames,
        late_entry_prob,
        late_entry_range,
        early_exit_prob,
        early_exit_range,
        feature_mask_prob,
        hazard_calibration_func,
        non_hazard_names,
    )
    _save_dataset(val_filename, *val_data, feature_names, non_hazard_names,
                  num_frames, late_entry_prob, feature_mask_prob,
                  hazard_calibration_func)

    if cal_filename is not None:
        if seed is not None:
            np.random.seed(seed + 2)
        print(
            f"Generating calibration set ({cal_num_cells} cells) -> {cal_filename}"
        )
        cal_data = _generate_cells(
            graph,
            cal_num_cells,
            num_frames,
            late_entry_prob,
            late_entry_range,
            early_exit_prob,
            early_exit_range,
            feature_mask_prob,
            hazard_calibration_func,
            non_hazard_names,
        )
        _save_dataset(cal_filename, *cal_data, feature_names, non_hazard_names,
                      num_frames, late_entry_prob, feature_mask_prob,
                      hazard_calibration_func)

    # print(f"Estimating variances")


    # hazard_branch_var, hazard_total_var, cdf_branch_var, cdf_total_var, bernoulli_floor = graph.get_variances(
    #     target_feature='Hazard',
    #     max_horizon=100,
    #     max_base_time_steps=num_frames,
    #     hazard_calibration_func=hazard_calibration_func,
    #     base_sample_size=200,
    #     branch_sample_size=1000,
    # )
    # for file in (val_filename, train_filename):
    #     with h5py.File(file, 'r+') as f:
    #         attrs = f['Cells']['Phase']['HazardRates'].attrs
    #         attrs['Total Variance'] = hazard_total_var
    #         attrs['Unobserved Variance'] = hazard_branch_var
    #         attrs['CDF Total Variance'] = cdf_total_var
    #         attrs['CDF Unobserved Variance'] = cdf_branch_var
    #         attrs['Bernoulli Floor'] = bernoulli_floor
def estimate_variances(
    graph: CausalGraph,
    hazard_calibration_func: callable,
    horizon: int,
    max_sequence_length: int,
    min_sequence_length: int,
    trunk_sample_size: int = 500,
    branch_smaple_size: int = 1000,
    hazard_bins: np.ndarray | None = None,
    max_time_to_death: int | None = None,
) -> dict[str, list[float]]:
    values = {'hazard': [], 'cdf': [], 'pmf': []}
    for _ in tqdm(range(trunk_sample_size), desc="Estimating Variances"):
        # Genrate full sample signal
        base_signals = graph.sample_graph(max_sequence_length)
        base_hazard = hazard_calibration_func(base_signals['Hazard'])

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
            continue
        lf = np.random.choice(lf)

        # Branch from landmark frame
        branch_signals = {}
        for feature in graph.features:
            trunk = base_signals[feature.name][:lf]
            extensions = np.stack([
                feature.generate_signal(horizon)
                for _ in range(branch_smaple_size)
            ],
                                  axis=1)
            branch_signals[feature.name] = np.vstack([
                np.broadcast_to(trunk[:, np.newaxis],
                                (lf, branch_smaple_size)),
                extensions,
            ])

        for t in range(lf, lf + horizon):
            for rule in graph.rules:
                rule.apply_step(branch_signals, t)

        # Get hazard of each branch
        branch_hazards = hazard_calibration_func(branch_signals['Hazard'][lf:])
        branch_cdfs = np.cumsum(_pmf_from_hazards(branch_hazards), axis=0)

        if hazard_bins is not None:
            h = np.stack([
                1.0 - np.prod(
                    1.0 - branch_hazards[hazard_bins[i]:hazard_bins[i + 1]],
                    axis=0) for i in range(len(hazard_bins) - 1)
            ])  # (n_bins, B)
        else:
            h = branch_hazards
        pmfs = _pmf_from_hazards(h)

        values['hazard'].append(h.T)  # (B, max_horizon_or_n_bins)
        values['cdf'].append(branch_cdfs.T)
        values['pmf'].append(pmfs.T)

        variances = {name: {} for name in values}

        for name, vals in values.items():
            vals = np.array(vals)
            branch_var = np.average(np.var(vals, axis=1), axis=0)
            total_var = branch_var + np.var(np.mean(vals, axis=1), axis=0)

            variances[name] = {'Total': total_var, 'Unobserved': branch_var}

    return variances


# def estimate_variances(
#     graph: CausalGraph,
#     hazard_calibration_func: callable,
#     max_horizon: int,
#     max_base_time_steps: int,
#     base_sample_size: int = 200,
#     branch_sample_size: int = 1000,
#     warm_up_steps: int = 200,
#     hazard_bins: np.ndarray | None = None,
#     landmark_distribution: np.ndarray | None = None,
# ):
#     """Estimate hazard and CDF variances by branching from sampled base trajectories.

#     If landmark_distribution is provided, base trajectory lengths are drawn from
#     that empirical distribution (e.g. the actual test-set landmark frames) rather
#     than uniformly from [warm_up_steps, warm_up_steps + max_base_time_steps].
#     This gives variance estimates that match the test-set landmark distribution exactly.
#     """
#     if hazard_bins is not None:
#         assert max_horizon >= hazard_bins[-1], (
#             f"max_horizon ({max_horizon}) must be >= hazard_bins[-1] ({hazard_bins[-1]})"
#         )

#     values = {'hazard': [], 'cdf': []}
#     base_lengths = []
#     base_survival = []
#     for _ in tqdm(range(base_sample_size), desc='Estimating variances'):
#         if landmark_distribution is not None:
#             base_time_steps = int(np.random.choice(landmark_distribution))
#         else:
#             base_time_steps = np.random.randint(
#                 warm_up_steps, warm_up_steps + max_base_time_steps)
#         base_signals = graph.sample_graph(base_time_steps)
#         base_hazard = hazard_calibration_func(base_signals['Hazard'])

#         # Probability this freshly-generated trajectory reaches the landmark alive.
#         base_lengths.append(base_time_steps)
#         base_survival.append(float(np.prod(1.0 - base_hazard)))

#         # Build batch signals: shape (base_time_steps + max_horizon, branch_sample_size)
#         # All branches share the base trajectory; each gets independent noise beyond it.
#         B = branch_sample_size
#         batch_signals = {}
#         for feature in graph.features:
#             base = base_signals[feature.name]  # (base_time_steps,)
#             extensions = np.stack(
#                 [feature.generate_signal(max_horizon) for _ in range(B)],
#                 axis=1,
#             )  # (max_horizon, B)
#             batch_signals[feature.name] = np.vstack([
#                 np.broadcast_to(base[:, np.newaxis], (base_time_steps, B)),
#                 extensions,
#             ])  # (base_time_steps + max_horizon, B)

#         # Apply rules over the horizon for all branches simultaneously
#         for t in range(base_time_steps, base_time_steps + max_horizon):
#             for rule in graph.rules:
#                 rule.apply_step(batch_signals, t)

#         # (max_horizon, B)
#         branch_hazards = hazard_calibration_func(
#             batch_signals['Hazard'][base_time_steps:])
#         branch_cdfs = np.cumsum(_pmf_from_hazards(branch_hazards),
#                                 axis=0)  # (max_horizon, B)

#         if hazard_bins is not None:
#             h = np.stack([
#                 1.0 - np.prod(
#                     1.0 - branch_hazards[hazard_bins[i]:hazard_bins[i + 1]],
#                     axis=0) for i in range(len(hazard_bins) - 1)
#             ])  # (n_bins, B)
#         else:
#             h = branch_hazards  # (max_horizon, B)

#         values['hazard'].append(h.T)  # (B, max_horizon_or_n_bins)
#         values['cdf'].append(branch_cdfs.T)  # (B, max_horizon)

#     # Importance weights correcting the freshly-generated base trajectories to the
#     # distribution of trajectories actually alive at the landmark.
#     base_lengths = np.array(base_lengths)
#     base_survival = np.array(base_survival)
#     if landmark_distribution is not None:
#         # Lengths already drawn from the test landmark distribution, so the length
#         # marginal needs no correction. Reweight trajectories by survival NORMALISED
#         # within each length (s / E[s|L]) so the weight only reshapes the trajectory
#         # mix at fixed L and leaves the sampled length marginal untouched.
#         weights = _within_length_survival_weights(base_lengths, base_survival)
#     else:
#         # Lengths drawn uniformly: the global raw survival weight simultaneously
#         # supplies the survival-shaped length marginal and the alive-at-L conditioning.
#         weights = base_survival

#     variances = {name: {} for name in values}

#     for name, vals in values.items():
#         # Calculate variance, weighting by base survival proabability (proabability such a sample would actaully appear in test set)
#         vals = np.array(vals)
#         branch_var = np.average(np.var(vals, axis=1), axis=0, weights=weights)
#         total_var = branch_var + weighted_var(
#             np.mean(vals, axis=1), axis=0,
#             weights=weights)  # By law of total variance

#         variances[name] = {'Total': total_var, 'Unobserved': branch_var}

#         # if name == 'cdf':
#         #     # Account for sampling error to allow cor comapision with MSE
#         #     base_av_p = np.mean(vals,
#         #                         axis=1)  # (base_sample_size, max_horizon)
#         #     av_p = np.average(base_av_p, axis=0, weights=branch_survival)
#         #     death_sampling_var = av_p * (1 - av_p) - weighted_var(
#         #         base_av_p, axis=0, weights=branch_survival)
#         #     variances['cdf']['Total'] += death_sampling_var
#         #     variances['cdf']['Unobserved'] += death_sampling_var

#     return variances


def _within_length_survival_weights(
    lengths: np.ndarray,
    survival: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Importance weight s / E[s|L] for conditioning on survival to the landmark.

    Reweights freshly-generated base trajectories to the population actually alive at
    their landmark, WITHOUT disturbing the (already-sampled) length marginal: survival
    is normalised by its mean within each length bin, so the weights average to 1 per
    bin. Lengths are binned by quantile so each bin holds a comparable sample count;
    E[s|L] needs several samples per bin, so keep n_bins << len(lengths).
    """
    lengths = np.asarray(lengths, dtype=float)
    survival = np.asarray(survival, dtype=float)
    n_bins = max(1, min(n_bins, len(lengths) // 5))

    edges = np.quantile(lengths, np.linspace(0.0, 1.0, n_bins + 1))
    bucket = np.clip(np.digitize(lengths, edges[1:-1]), 0, n_bins - 1)

    weights = np.ones_like(survival)
    for b in np.unique(bucket):
        mask = bucket == b
        mean_s = survival[mask].mean()
        if mean_s > 0:
            weights[mask] = survival[mask] / mean_s
    return weights


def weighted_var(arr: np.ndarray,
                 weights: np.ndarray,
                 axis: int = None) -> np.ndarray:
    """Weighted variance along axis, analogous to np.var(arr, axis=axis)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    if axis is None:
        mean = np.dot(w.ravel(), arr.ravel())
        return np.dot(w.ravel(), (arr.ravel() - mean)**2)

    shape = [1] * arr.ndim
    shape[axis] = -1
    w = w.reshape(shape)

    mean = np.sum(w * arr, axis=axis, keepdims=True)
    return np.sum(w * (arr - mean)**2, axis=axis)


# def save_variances(file_names: list[Path],
#                    graph: CausalGraph,
#                    hazard_calibration_func: callable,
#                    max_horizon: int,
#                    max_base_time_steps: int,
#                    base_sample_size: int = 200,
#                    branch_sample_size: int = 1000,
#                    hazard_bins: np.ndarray | None = None):
#     variances = _estimate_variances(
#         graph,
#         hazard_calibration_func,
#         max_horizon,
#         max_base_time_steps,
#         base_sample_size,
#         branch_sample_size,
#         hazard_bins=hazard_bins,
#     )
#     dataset_map = {'hazard': 'HazardRates', 'cdf': 'CIFs'}
#     for file_name in file_names:
#         with h5py.File(file_name, 'r+') as f:
#             for name, dataset_name in dataset_map.items():
#                 attrs = f['Cells']['Phase'][dataset_name].attrs
#                 attrs['Total Variance'] = variances[name]['Total']
#                 attrs['Unobserved Variance'] = variances[name]['Unobserved']
#             if hazard_bins is not None:
#                 f['Cells']['Phase']['HazardRates'].attrs[
#                     'Hazard Bins'] = hazard_bins

# _QUANTITY_DATASET = {'hazard': 'HazardRates', 'cdf': 'CIFs'}

# def plot_variances_on_ax(
#         file_path: Path,
#         ax: plt.Axes,
#         colour: str | tuple,
#         label: str,
#         quantity: Literal['hazard', 'cdf'] = 'hazard') -> None:
#     """Plot Total and Unobserved variance for one quantity ('hazard' or 'cdf')."""
#     dataset_name = _QUANTITY_DATASET[quantity]
#     with h5py.File(file_path, 'r') as f:
#         attrs = f['Cells']['Phase'][dataset_name].attrs
#         total = attrs['Total Variance']
#         unobserved = attrs['Unobserved Variance']
#         bins = attrs.get('Hazard Bins') if quantity == 'hazard' else None

#     horizons = bins[:-1] if bins is not None else np.arange(len(total))
#     ax.plot(horizons,
#             total,
#             marker='.',
#             markersize=3,
#             linestyle='dotted',
#             label=f'{label} Total',
#             color=colour)
#     ax.plot(horizons,
#             unobserved,
#             marker='.',
#             markersize=3,
#             label=f'{label} Unobserved',
#             color=colour)
