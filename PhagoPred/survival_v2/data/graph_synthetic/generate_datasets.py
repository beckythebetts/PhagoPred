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
    smooth_hazard: bool,
    smooth_sigma: int,
    multiplier: float,
    n_samples: int,
) -> float:
    """Estimate the death fraction for a given hazard multiplier over n_samples cells."""
    deaths = 0
    for c in range(n_samples):
        signals = graph.sample_graph()
        hazard_signal = signals['Hazard'].copy()
        if smooth_hazard:
            shift = 2 * smooth_sigma
            kernel_len = shift + int(4 * smooth_sigma) + 1
            t_k = np.arange(kernel_len)
            kernel = np.exp(-0.5 * ((t_k - shift) / smooth_sigma)**2)
            kernel /= kernel.sum()
            smoothed = np.convolve(hazard_signal, kernel,
                                   mode='full')[:num_frames]
        else:
            smoothed = hazard_signal.copy()
        smoothed -= smoothed.min()
        mx = smoothed.max()
        if mx > 0:
            smoothed /= mx
        # Apply multiplier after normalisation so it actually scales hazard rates
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
    smooth_hazard: bool,
    smooth_sigma: int,
    target_death_fraction: float,
    n_calib: int = 200,
    max_iter: int = 50,
    tol: float = 0.02,
) -> float:
    """Iteratively find a post-normalisation scalar multiplier on hazard rates
    so that approximately target_death_fraction of cells die within their window.

    Each iteration estimates the death fraction at the current multiplier, then
    updates via multiplier *= target / observed (multiplicative Newton step).
    Stops when within `tol` of the target or after `max_iter` iterations.
    """
    multiplier = 1.0
    for i in range(max_iter):
        observed = _observe_death_fraction(
            graph,
            num_frames,
            start_frames,
            end_frames,
            smooth_hazard,
            smooth_sigma,
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
        # multiplier > 1 is fine — values are clipped to [0,1] after scaling
    return float(multiplier)


def generate_dataset(
    filename: Path,
    features: list[Feature],
    rules: list[Rule],
    num_cells: int = 1000,
    num_frames: int = 1000,
    late_entry_prob: float = 0.0,
    late_entry_range: tuple = (0, 100),
    feature_mask_prob: float = 0.0,
    smooth_hazard: bool = True,
    smooth_sigma: int = 10,
    target_death_fraction: float = None,
    seed: int = None,
) -> None:
    """Generate a synthetic survival dataset from a CausalGraph and save to HDF5.

    Args:
        filename: Output .h5 file path.
        features: List of Feature objects (must include one named 'Hazard').
        rules: List of Rule objects defining the causal graph.
        num_cells: Number of cell trajectories to generate.
        num_frames: Number of time frames per trajectory.
        late_entry_prob: Fraction of cells with a late observation start
            (truncation). 0 disables late entry entirely.
        late_entry_range: (min, max) frame range for late entry start times.
        feature_mask_prob: Per-frame probability of masking feature values
            with NaN (applied after observation window). 0 disables masking.
        smooth_hazard: If True, apply a forward-shifted Gaussian smooth to the
            Hazard signal before converting to hazard rates.
        smooth_sigma: Gaussian sigma (frames) used when smooth_hazard is True.
        target_death_fraction: If set, run a calibration pass to find a scalar
            multiplier on the Hazard signal so that approximately this fraction
            of cells die within their observation window.
        seed: Random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    feature_names = [f.name for f in features]
    assert 'Hazard' in feature_names, "Features must include a node named 'Hazard'."

    graph = CausalGraph(features, rules, time_steps=num_frames)

    # Observation windows per cell
    start_frames = np.zeros(num_cells, dtype=int)
    if late_entry_prob > 0.0:
        late_mask = np.random.rand(num_cells) < late_entry_prob
        start_frames[late_mask] = np.random.randint(late_entry_range[0],
                                                    late_entry_range[1],
                                                    size=int(late_mask.sum()))
    else:
        start_frames[:] = np.random.randint(0, 10, size=num_cells)

    end_frames = np.random.randint(num_frames // 2, num_frames, size=num_cells)

    # Calibrate hazard multiplier to hit target_death_fraction
    if target_death_fraction is not None:
        calib_state = np.random.get_state()
        np.random.seed((seed or 0) + 99999)
        calib_starts = np.random.randint(0, 10, size=200)
        calib_ends = np.random.randint(num_frames // 2, num_frames, size=200)
        hazard_multiplier = _calibrate_multiplier(
            graph,
            num_frames,
            calib_starts,
            calib_ends,
            smooth_hazard,
            smooth_sigma,
            target_death_fraction,
        )
        np.random.set_state(calib_state)
    else:
        hazard_multiplier = 1.0

    # Pre-allocate storage
    non_hazard_names = [n for n in feature_names if n != 'Hazard']
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

        # --- derive hazard rates from the 'Hazard' feature ---
        hazard_signal = signals['Hazard'].copy()

        if smooth_hazard:
            shift = 2 * smooth_sigma
            kernel_len = shift + int(4 * smooth_sigma) + 1
            t_k = np.arange(kernel_len)
            kernel = np.exp(-0.5 * ((t_k - shift) / smooth_sigma)**2)
            kernel /= kernel.sum()
            smoothed = np.convolve(hazard_signal, kernel,
                                   mode='full')[:num_frames]
        else:
            smoothed = hazard_signal.copy()

        # Normalise to [0, 1]: shift to non-negative then scale
        smoothed -= smoothed.min()
        max_val = smoothed.max()
        if max_val > 0:
            smoothed /= max_val
        hazard_rates = np.clip(smoothed * hazard_multiplier, 0.0,
                               1.0).astype(np.float32)

        pmf = _pmf_from_hazards(hazard_rates).astype(np.float32)
        cif = np.cumsum(pmf).astype(np.float32)

        all_hazard_rates[:, c] = hazard_rates
        all_pmfs[:, c] = pmf
        all_cifs[:, c] = cif

        # --- sample death frame ---
        u = np.random.rand()
        if u > cif.max():
            death_frame = np.nan
        else:
            death_frame = float(np.argmax(cif >= u))
            if death_frame < start or death_frame > end:
                death_frame = np.nan

        all_deaths[c] = death_frame

        # --- apply observation window to features ---
        for name in non_hazard_names:
            sig = signals[name].astype(np.float32)
            sig[:start] = np.nan
            sig[end + 1:] = np.nan
            if feature_mask_prob > 0.0:
                mask = np.random.rand(num_frames) < feature_mask_prob
                sig[mask] = np.nan
            all_features[name][:, c] = sig

        # Store the raw Hazard signal (within observation window only)
        hz_sig = hazard_signal.astype(np.float32)
        hz_sig[:start] = np.nan
        hz_sig[end + 1:] = np.nan
        all_hazard_signals[:, c] = hz_sig

    # --- save to HDF5 ---
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

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
