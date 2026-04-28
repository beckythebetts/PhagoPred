from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .graph import CausalGraph, Feature
from .rules import Rule


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    sf = np.cumprod(1.0 - hazards)
    pmf = np.zeros(len(hazards))
    pmf[0] = hazards[0]
    for t in range(1, len(hazards)):
        pmf[t] = sf[t - 1] * hazards[t]
    return pmf


def _observe_death_fraction(
    graph: CausalGraph,
    num_frames: int,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    # smooth_hazard: bool,
    # smooth_sigma: int,
    multiplier: float,
    n_samples: int,
) -> float:
    """Estimate the death fraction for a given hazard multiplier over n_samples cells."""
    deaths = 0
    for c in range(n_samples):
        signals = graph.sample_graph()
        hazard_signal = signals['Hazard'].copy()
        # if smooth_hazard:
        #     shift = 2 * smooth_sigma
        #     kernel_len = shift + int(4 * smooth_sigma) + 1
        #     t_k = np.arange(kernel_len)
        #     kernel = np.exp(-0.5 * ((t_k - shift) / smooth_sigma)**2)
        #     kernel /= kernel.sum()
        #     smoothed = np.convolve(hazard_signal, kernel,
        #                            mode='full')[:num_frames]
        # else:
        smoothed = hazard_signal.copy()
        hazard_rates = np.clip(smoothed * multiplier, 0.0, 1.0)
        pmf = _pmf_from_hazards(hazard_rates)
        cif = np.cumsum(pmf)
        start, end = int(start_frames[c]), int(end_frames[c])
        u = np.random.rand()
        if u <= cif.max():
            df = float(np.argmax(cif >= u))
            if start <= df <= end:
                deaths += 1
    return deaths / n_samples


def _calibrate_multiplier(
    graph: CausalGraph,
    num_frames: int,
    start_frames: np.ndarray,
    end_frames: np.ndarray,
    # smooth_hazard: bool,
    # smooth_sigma: int,
    target_death_fraction: float,
    n_calib: int = 200,
    max_iter: int = 50,
    tol: float = 0.02,
) -> float:
    """Iteratively find a post-normalisation scalar multiplier on hazard rates
    so that approximately target_death_fraction of cells die within their window."""
    multiplier = 1e-4
    for i in range(max_iter):
        observed = _observe_death_fraction(
            graph,
            num_frames,
            start_frames,
            end_frames,
            # smooth_hazard,
            # smooth_sigma,
            multiplier=multiplier,
            n_samples=n_calib,
        )
        print(f"  Calibration iter {i + 1}: multiplier={multiplier:.4f}, "
              f"observed={observed:.3f}, target={target_death_fraction:.3f}")
        if abs(observed - target_death_fraction) <= tol:
            break
        if observed < 1e-6:
            multiplier *= 10.0
        else:
            multiplier *= target_death_fraction / observed
    return float(multiplier)


def _generate_cells(
    graph: CausalGraph,
    num_cells: int,
    num_frames: int,
    late_entry_prob: float,
    late_entry_range: tuple,
    early_exit_prob: float,
    early_exit_range: tuple,
    feature_mask_prob: float,
    # smooth_hazard: bool,
    # smooth_sigma: int,
    hazard_multiplier: float,
    feature_names: list,
    non_hazard_names: list,
) -> tuple:
    """Generate cell trajectories using a pre-calibrated hazard multiplier."""
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
    # end_frames = np.random.randint(num_frames // 2, num_frames, size=num_cells)

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

        # if smooth_hazard:
        #     shift = 2 * smooth_sigma
        #     kernel_len = shift + int(4 * smooth_sigma) + 1
        #     t_k = np.arange(kernel_len)
        #     kernel = np.exp(-0.5 * ((t_k - shift) / smooth_sigma)**2)
        #     kernel /= kernel.sum()
        #     smoothed = np.convolve(hazard_signal, kernel,
        #                            mode='full')[:num_frames]
        # else:
        smoothed = hazard_signal.copy()

        hazard_rates = np.clip(smoothed * hazard_multiplier, 0.0,
                               1.0).astype(np.float32)

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

        meta = f.require_group('Metadata')
        meta.attrs['num_cells'] = num_cells
        meta.attrs['num_frames'] = num_frames
        meta.attrs['late_entry_prob'] = late_entry_prob
        meta.attrs['feature_mask_prob'] = feature_mask_prob
        meta.attrs['feature_names'] = feature_names


def generate_dataset(
    train_filename: Path,
    val_filename: Path,
    features: list[Feature],
    rules: list[Rule],
    train_num_cells: int = 1000,
    val_num_cells: int = 200,
    num_frames: int = 500,
    late_entry_prob: float = 0.0,
    late_entry_range: tuple = (0, 100),
    early_exit_prob: float = 0.0,
    early_exit_range: float = (0, 100),
    feature_mask_prob: float = 0.0,
    target_death_fraction: float = 0.5,
    seed: int = None,
) -> None:
    """Generate train and validation synthetic survival datasets from a CausalGraph.

    Calibrates the hazard multiplier once (from target_death_fraction) then
    uses it for both splits, ensuring they share the same hazard scaling.

    Args:
        train_filename: Output .h5 path for the training set.
        val_filename: Output .h5 path for the validation set.
        features: List of Feature objects (must include one named 'Hazard').
        rules: List of Rule objects defining the causal graph.
        train_num_cells: Number of cells in the training set.
        val_num_cells: Number of cells in the validation set.
        num_frames: Number of time frames per trajectory.
        late_entry_prob: Fraction of cells with a late observation start.
        late_entry_range: (min, max) frame range for late entry start times.
        feature_mask_prob: Per-frame probability of masking feature values with NaN.
        smooth_hazard: If True, apply a forward-shifted Gaussian smooth to Hazard.
        smooth_sigma: Gaussian sigma (frames) used when smooth_hazard is True.
        target_death_fraction: Calibrate hazard multiplier so this fraction of
            cells die within their observation window.
        seed: Random seed for reproducibility. Val set uses seed+1.
    """
    if seed is not None:
        np.random.seed(seed)

    feature_names = [f.name for f in features]
    assert 'Hazard' in feature_names, "Features must include a node named 'Hazard'."
    non_hazard_names = [n for n in feature_names if n != 'Hazard']

    graph = CausalGraph(features, rules, time_steps=num_frames)

    # Calibrate multiplier once using a separate rng state so it doesn't
    # affect the reproducibility of the train/val splits.
    calib_state = np.random.get_state()
    np.random.seed((seed or 0) + 99999)
    calib_starts = np.random.randint(0, 10, size=200)
    calib_ends = np.random.randint(num_frames - 10, num_frames, size=200)
    hazard_multiplier = _calibrate_multiplier(
        graph,
        num_frames,
        calib_starts,
        calib_ends,
        target_death_fraction,
    )
    np.random.set_state(calib_state)

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
        hazard_multiplier,
        feature_names,
        non_hazard_names,
    )
    _save_dataset(train_filename, *train_data, feature_names, non_hazard_names,
                  num_frames, late_entry_prob, feature_mask_prob)

    # Val set uses a different seed so cells are independent of the train set.
    if seed is not None:
        np.random.seed(seed + 1)
    print(
        f"Generating validation set ({val_num_cells} cells) -> {val_filename}")
    val_data = _generate_cells(
        graph,
        train_num_cells,
        num_frames,
        late_entry_prob,
        late_entry_range,
        early_exit_prob,
        early_exit_range,
        feature_mask_prob,
        hazard_multiplier,
        feature_names,
        non_hazard_names,
    )
    _save_dataset(val_filename, *val_data, feature_names, non_hazard_names,
                  num_frames, late_entry_prob, feature_mask_prob)
