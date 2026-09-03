"""Reading and aggregating SHAP importance maps from a shap_samples.h5 file.

Pure numpy/h5py so both the per-experiment plots in ``ground_truth_importance``
and the cross-model plots in ``experiments.plots`` can share it without pulling
in torch.

Three grids are in play:

* ground truth importances are per (feature, frame) over ``[0, lf)``
* model KernelSHAP is per (segment, model feature), 50 segments spanning
  ``[0, lf)``
* samples have different ``lf``

``spread_segments`` puts the model on the frame grid by dividing each segment's
attribution by its length and repeating it across the segment's frames, so the
per-frame values are densities and the total attribution is preserved.
``DatasetAverage`` then stacks samples on an absolute frame axis, NaN-padding
short ones, so column ``j`` averages only the samples whose ``lf`` reaches it.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import numpy as np

GROUND_TRUTH_KEYS = ('Interventional', 'Observational')


def as_str(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def spread_segments(values: np.ndarray,
                    boundaries: np.ndarray) -> np.ndarray:
    """``(..., n_segments)`` -> ``(..., lf)`` per-frame attribution density.

    Each segment's value is divided by its frame count and repeated across those
    frames, so summing the result over frames recovers the original total.
    """
    lengths = np.diff(np.asarray(boundaries, dtype=int))
    values = np.asarray(values, dtype=float)
    density = values / np.where(lengths == 0, 1, lengths)
    return np.repeat(density, lengths, axis=-1)


def horizon_outputs(
    horizon_hazard: np.ndarray | None,
    output_type: str,
    hazard_bins: np.ndarray | None = None,
) -> float | np.ndarray | None:
    """The ground-truth prediction target, derived from the realised hazard.

    ``horizon_hazard`` is the calibrated hazard along the realised trajectory
    over ``[lf, lf + horizon)``. Binary models predict P(event within horizon);
    survival models predict a per-bin PMF. Both fall out of the survival
    function ``S(t) = prod_{u < t} (1 - h_u)``::

        CIF(horizon) = 1 - S(horizon)
        pmf[i]       = S(bins[i]) - S(bins[i + 1])

    This reproduces ``analytic_estimates._outputs_from_hazard``'s ``binary_prob``
    and ``pmf`` without importing it, keeping the plotting stack free of the
    causal-graph machinery (graphviz, pyvis, networkx).

    Returns None when no hazard was stored, so callers can degrade gracefully.
    """
    if horizon_hazard is None:
        return None
    hazard = np.clip(np.asarray(horizon_hazard, dtype=float), 0.0, 1.0)
    # survival[t] = P(survive frames 0 .. t-1); survival[0] = 1 by definition
    survival = np.concatenate([[1.0], np.cumprod(1.0 - hazard)])

    if output_type == 'binary':
        return float(1.0 - survival[-1])

    if hazard_bins is None:
        raise ValueError("survival output_type needs hazard bins")
    edges = np.asarray(hazard_bins, dtype=int)
    if edges[-1] > len(hazard):
        raise ValueError(f'hazard bins reach frame {edges[-1]} but only '
                         f'{len(hazard)} hazard frames were stored')
    return survival[edges[:-1]] - survival[edges[1:]]


def select_rows(importance_map: np.ndarray, names: list[str],
                keep: list[str]) -> np.ndarray:
    """Reorder/subset an importance map's feature rows to ``keep``."""
    index = {name: i for i, name in enumerate(names)}
    missing = [k for k in keep if k not in index]
    if missing:
        raise KeyError(f'features {missing} not among {names}')
    return np.asarray(importance_map, dtype=float)[[index[k] for k in keep]]


def temporal_profile(importance_map: np.ndarray) -> np.ndarray:
    """(features, frames) -> (frames,), summed over features."""
    return np.asarray(importance_map, dtype=float).sum(axis=0)


def feature_profile(importance_map: np.ndarray) -> np.ndarray:
    """(features, frames) -> (features,), summed over frames."""
    return np.asarray(importance_map, dtype=float).sum(axis=1)


def pad_frames(array: np.ndarray, max_frames: int) -> np.ndarray:
    """NaN-pad the trailing frame axis out to ``max_frames``."""
    array = np.asarray(array, dtype=float)
    pad = max_frames - array.shape[-1]
    if pad <= 0:
        return array
    widths = [(0, 0)] * (array.ndim - 1) + [(0, pad)]
    return np.pad(array, widths, constant_values=np.nan)


def _nanmean(stack: np.ndarray, axis: int = 0) -> np.ndarray:
    """nanmean that returns NaN (not a RuntimeWarning) for all-NaN slices."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, axis=axis)


@dataclass
class SampleImportances:
    """Everything plottable for one stored sample, on the frame grid."""
    index: int | None
    landmark_frame: int
    death_frame: float | None
    feature_names: list[str]
    signals: np.ndarray | None  # (n_features, lf)
    ground_truth: dict[str, np.ndarray]  # name -> (n_features, lf) joint 2D map
    # Ground-truth temporal/feature marginals computed as *separate* Shapley
    # games (not projections of the 2D map), one per estimator. Empty if the
    # sample predates the three-axis estimator, in which case the accessors fall
    # back to projecting the 2D map.
    ground_truth_temporal_game: dict[str, np.ndarray]  # name -> (lf,) density
    ground_truth_feature_game: dict[str, np.ndarray]  # name -> (n_features,)
    model_feature_names: list[str] | None
    model_map: np.ndarray | None  # (n_model_features, lf) density
    # KernelSHAP's separately computed marginals (its own temporal/feature games).
    model_temporal: np.ndarray | None  # (lf,) density, separate game
    model_feature: np.ndarray | None  # (n_model_features,) separate game
    segment_boundaries: np.ndarray | None
    horizon_hazard: np.ndarray | None
    model_prediction: np.ndarray | None

    @property
    def shared_feature_names(self) -> list[str]:
        """Features the model actually sees (the graph's Hazard node is hidden).

        Temporal profiles are summed over these only, so the ground-truth and
        model curves are sums over the same set. Hazard still appears in the
        heatmaps and the per-feature bars, where it is not being summed against
        anything.
        """
        return (self.feature_names
                if self.model_feature_names is None else
                self.model_feature_names)

    def ground_truth_temporal(self, key: str) -> np.ndarray:
        """The temporal *game* if stored, else the 2D map projected over features."""
        stored = self.ground_truth_temporal_game.get(key)
        if stored is not None:
            return np.asarray(stored, dtype=float)
        return temporal_profile(
            select_rows(self.ground_truth[key], self.feature_names,
                        self.shared_feature_names))

    def ground_truth_feature(self, key: str) -> np.ndarray:
        """The feature *game* if stored, else the 2D map projected over frames."""
        stored = self.ground_truth_feature_game.get(key)
        if stored is not None:
            return np.asarray(stored, dtype=float)
        return feature_profile(self.ground_truth[key])

    def model_temporal_from_map(self) -> np.ndarray | None:
        """The model's temporal game if stored, else the joint map over features."""
        if self.model_temporal is not None:
            return np.asarray(self.model_temporal, dtype=float)
        return None if self.model_map is None else temporal_profile(
            self.model_map)

    def model_feature_from_map(self) -> np.ndarray | None:
        """The model's feature game if stored, else the joint map over frames."""
        if self.model_feature is not None:
            return np.asarray(self.model_feature, dtype=float)
        return None if self.model_map is None else feature_profile(
            self.model_map)


def read_root_attrs(h5_path: Path | str) -> dict:
    with h5py.File(h5_path, 'r') as f:
        bins = np.asarray(f.attrs['Hazard Bins'])
        return {
            'scenario': as_str(f.attrs['Scenario']),
            'output_type': as_str(f.attrs['Output type']),
            'horizon': int(f.attrs['Horizon']),
            'hazard_bins': None if bins.size == 0 else bins,
            'num_permutations': int(f.attrs['Num Permutations']),
        }


def read_root_attrs_optional(h5_path: Path | str) -> dict | None:
    """``read_root_attrs`` for files that may not carry the ground-truth root
    metadata.

    ``run_interpret`` writes a ``SHAP.h5`` with per-sample model SHAP only and no
    scenario / horizon / hazard-bin attrs at the root. Returns None in that case
    so the plotting stack can drop the outputs panel rather than raise.
    """
    try:
        return read_root_attrs(h5_path)
    except (KeyError, OSError):
        return None


def sample_indices(h5_path: Path | str) -> list[int]:
    """Numerically sorted sample ids (h5py iterates keys lexically: 0, 1, 10...)."""
    with h5py.File(h5_path, 'r') as f:
        if 'Signals' in f:
            return []  # single-sample file written at the root
        return sorted(int(k) for k in f.keys())


def load_sample_importances(h5_path: Path | str,
                            sample_idx: int | None = None
                            ) -> SampleImportances:
    with h5py.File(h5_path, 'r') as f:
        group = f[str(sample_idx)] if sample_idx is not None else f

        feature_names = [as_str(n) for n in group.attrs['Features']]
        lf = int(group.attrs['Landmark Frame'])
        death_frame = group.attrs.get('Death Frame', None)
        death_frame = None if death_frame is None else float(death_frame)

        signals = group['Signals'][:, :lf] if 'Signals' in group else None
        ground_truth = {}
        ground_truth_temporal_game = {}
        ground_truth_feature_game = {}
        for key in GROUND_TRUTH_KEYS:
            if key not in group:
                continue
            ds = group[key]
            ground_truth[key] = ds[:]
            if 'Temporal' in ds.attrs:  # separate temporal game (lf,) density
                ground_truth_temporal_game[key] = np.asarray(ds.attrs['Temporal'])
            if 'Feature' in ds.attrs:  # separate feature game (n_features,)
                ground_truth_feature_game[key] = np.asarray(ds.attrs['Feature'])
        horizon_hazard = (group['Horizon Hazard'][:]
                          if 'Horizon Hazard' in group else None)
        model_prediction = (group['Model Prediction'][:]
                            if 'Model Prediction' in group else None)

        model_map = model_temporal = model_feature = None
        model_feature_names = segment_boundaries = None
        if 'Model' in group:
            dataset = group['Model']
            segment_boundaries = np.asarray(
                dataset.attrs['Segment Boundaries'], dtype=int)
            model_feature_names = [
                as_str(n) for n in dataset.attrs['Feature Names']
            ]
            # stored (n_segments, n_model_features)
            model_map = spread_segments(dataset[:].T, segment_boundaries)
            model_temporal = spread_segments(
                np.asarray(dataset.attrs['Temporal']), segment_boundaries)
            model_feature = np.asarray(dataset.attrs['Feature'], dtype=float)

    return SampleImportances(
        index=sample_idx,
        landmark_frame=lf,
        death_frame=death_frame,
        feature_names=feature_names,
        signals=signals,
        ground_truth=ground_truth,
        ground_truth_temporal_game=ground_truth_temporal_game,
        ground_truth_feature_game=ground_truth_feature_game,
        model_feature_names=model_feature_names,
        model_map=model_map,
        model_temporal=model_temporal,
        model_feature=model_feature,
        segment_boundaries=segment_boundaries,
        horizon_hazard=horizon_hazard,
        model_prediction=model_prediction,
    )


@dataclass
class DatasetAverage:
    """Magnitude (|.|) averages over every sample, on an absolute frame axis.

    Short samples are NaN-padded, so ``support[j]`` records how many samples
    actually reach frame ``j`` — the right-hand columns are averages over the
    few long-``lf`` samples and should be read with that in mind.
    """
    num_samples: int
    max_landmark_frame: int
    support: np.ndarray  # (max_lf,) samples contributing to each frame
    feature_names: list[str]
    model_feature_names: list[str] | None
    maps: dict[str, np.ndarray]  # name -> (n_features, max_lf)
    temporal: dict[str, np.ndarray]  # name -> (max_lf,)
    feature: dict[str, np.ndarray]  # name -> (n_features,)


def dataset_average(h5_path: Path | str,
                    indices: list[int] | None = None) -> DatasetAverage:
    """Mean |SHAP| across samples for every map, temporal and feature profile.

    Keys are the ground-truth estimator names plus ``'Model'`` (collapsed from
    the joint segment x feature map) when model SHAP is present.
    """
    if indices is None:
        indices = sample_indices(h5_path)
    if not indices:
        raise ValueError(f'No samples found in {h5_path}')

    samples = [load_sample_importances(h5_path, i) for i in indices]
    max_lf = max(s.landmark_frame for s in samples)
    feature_names = samples[0].feature_names
    model_feature_names = samples[0].model_feature_names

    support = np.zeros(max_lf, dtype=int)
    for s in samples:
        support[:s.landmark_frame] += 1

    maps: dict[str, list[np.ndarray]] = {}
    temporal: dict[str, list[np.ndarray]] = {}
    feature: dict[str, list[np.ndarray]] = {}

    def add(store: dict, key: str, value: np.ndarray) -> None:
        store.setdefault(key, []).append(value)

    for s in samples:
        for key, gt_map in s.ground_truth.items():
            add(maps, key, pad_frames(np.abs(gt_map), max_lf))
            # temporal and feature are the *separate* Shapley games (falling back
            # to projecting the 2D map for samples that predate them).
            add(temporal, key,
                pad_frames(np.abs(s.ground_truth_temporal(key)), max_lf))
            add(feature, key, np.abs(s.ground_truth_feature(key)))
        if s.model_map is not None:
            add(maps, 'Model', pad_frames(np.abs(s.model_map), max_lf))
            add(temporal, 'Model',
                pad_frames(np.abs(s.model_temporal_from_map()), max_lf))
            add(feature, 'Model', np.abs(s.model_feature_from_map()))

    return DatasetAverage(
        num_samples=len(samples),
        max_landmark_frame=max_lf,
        support=support,
        feature_names=feature_names,
        model_feature_names=model_feature_names,
        maps={k: _nanmean(np.stack(v)) for k, v in maps.items()},
        temporal={k: _nanmean(np.stack(v)) for k, v in temporal.items()},
        feature={k: np.mean(np.stack(v), axis=0) for k, v in feature.items()},
    )
