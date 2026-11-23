from __future__ import annotations
from pathlib import Path
import threading
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal
from functools import cached_property

import torch
import h5py
import numpy as np

from PhagoPred.utils.logger import get_logger
from .graph_synthetic.scenarios import load_scenario
from .graph_synthetic.variances import estimate_unobserved_variance, Variances

log = get_logger()

ArrayType_T = TypeVar(  # pylint: disable=invalid-name
    'ArrayType_T')  # np.ndarry for single sample, torch.Tensor for batch
ScalarType_T = TypeVar(  # pylint: disable=invalid-name
    'ScalarType_T')  # int/float for single sample, torch.Tensor for batch
PathType_T = TypeVar(  # pylint: disable=invalid-name
    'PathType_T')  # Path for single smaple of list[Path] for batch
ListType_T = TypeVar(  # pylint: disable=invalid-name
    'ListType_T')  # list for single sample, list[list] fo batch


@dataclass
class CellSample(ABC, Generic[ArrayType_T, ScalarType_T, PathType_T,
                              ListType_T]):
    """Abstract data class for single sample or batch from CellDataset."""
    features: ArrayType_T  # (num_frames, num_features)
    mask: ArrayType_T  # (num_frames, )
    length: ScalarType_T
    cell_idx: ScalarType_T
    hdf5_path: PathType_T
    start_frame: ScalarType_T
    landmark_frame: ScalarType_T
    last_frame: ScalarType_T


@dataclass
class CellDatasetMetadata:
    """Data class to hold metadata about the cells in a CellDataset."""
    file_idxs: list = field(default_factory=list)
    local_cell_idxs: list = field(default_factory=list)
    death_frames: list = field(default_factory=list)
    start_frames: list = field(default_factory=list)
    end_frames: list = field(default_factory=list)
    landmark_frames: list = field(default_factory=list)

    landmarks_populated: bool = False

    def get_cell(self, idx: int) -> CellDatasetMetadata:
        """Get metadata for given cell index."""
        cell = CellDatasetMetadata()
        for f in fields(self):
            if not isinstance(getattr(self, f.name), list):
                continue
            if f.name == 'landmark_frames' and not self.landmarks_populated:
                continue
            setattr(cell, f.name, getattr(self, f.name)[idx])
        return cell

    def apply_mask(self, mask: np.ndarray) -> None:
        """Apply boolean mask to all metadata fields."""
        for f in fields(self):
            if not isinstance(getattr(self, f.name), list):
                continue
            setattr(self, f.name,
                    list(np.array(getattr(self, f.name), dtype=object)[mask]))

    def __len__(self):
        return len(self.file_idxs)


class CellDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for pytorch datasets using .h5 files."""

    def __init__(
        self,
        hdf5_paths: list[Path] | list[str],
        feature_names: list[str],
        min_length: int = 50,
        means: np.ndarray | None | list = None,
        stds: np.ndarray | None | list = None,
        model_type: Literal['deep', 'classical'] = 'deep',
        num_cells: int | None = None,
    ):
        """
        Pytorch dataset using .h5 files.
        Args
        ----   
            hdf5_paths: List of paths to .h5 files containing cell data.
            feature_names: List of feature names to load from the .h5 files.
            means: Optional array of feature means for normalisation.
            stds: Optional array of feature stds for normalisation.
        """
        self.hdf5_paths = [Path(path) for path in hdf5_paths]
        self.feature_names = feature_names
        self.min_length = min_length
        self.means = np.array(means) if means is not None else None
        self.stds = np.array(stds) if stds is not None else None
        self.model_type = model_type
        self.num_cells = num_cells

        self._clean_features_cache = [None] * len(hdf5_paths)
        self._raw_context_cache = [None] * len(hdf5_paths)
        self._lock = threading.Lock()
        self._files = [None] * len(hdf5_paths)

        self.full_len_frames = 0

        self._features_cache = [None] * len(hdf5_paths)
        self._pmfs_cache = [None] * len(hdf5_paths)

        self.cell_metadata = CellDatasetMetadata()

        self.total_variances = None
        self.unobserved_variances = None

        self._load_features_cache()
        self._load_cell_metadata()
        self._remove_invalid_samples()

        self._close_files()

    def _get_landmark_weights(self) -> np.ndarray:
        return np.array([len(lf) for lf in self.cell_metadata.landmark_frames],
                        dtype=float)

    def make_sampler(
        self,
        num_samples: int | None = None
    ) -> torch.utils.data.WeightedRandomSampler:
        weights = self._get_landmark_weights()
        num_samples = num_samples if num_samples is not None else len(self)
        return torch.utils.data.WeightedRandomSampler(weights,
                                                      num_samples,
                                                      replacement=True)

    def sample_indices(self, num_samples: int | None = None) -> np.ndarray:
        weights = self._get_landmark_weights()
        weights = weights / weights.sum()
        idxs = np.random.choice(np.arange(len(self)), num_samples, p=weights)
        return idxs

    def _load_features_cache(self) -> None:
        """Load features from all .h5 files into memory."""
        for i, _ in enumerate(self.hdf5_paths):
            features_cache = self._get_features_np(i)
            self._features_cache[
                i] = features_cache  # list[(num_frames, num_cells)]

            self.full_len_frames = max(self.full_len_frames,
                                       features_cache.shape[1])

            # Synthetic files only: true PMFs live on Cells, not Cells/Phase.
            cells = self._get_file(i)['Cells']
            if 'PMFs' in cells:
                self._pmfs_cache[i] = cells['PMFs'][:, :self.num_cells]
                # list[(num_frames, num_cells)]

    def _get_features_np(self,
                         file_idx: int,
                         features: list = None) -> np.ndarray:
        """Load specified features from the given .h5 file index as a NumPy array.
        Returns
        -------
            features_array [nm_fatures, num_frames, num_cells]
        """
        if features is None:
            features = self.feature_names
        feature_arrays = []
        for feature in features:
            f = self._get_file(file_idx)
            arr = f['Cells']['Phase'][
                feature][:]  # shape: (num_frame, num_cells)
            if self.num_cells is not None and self.num_cells < arr.shape[1]:
                arr = arr[:, :self.num_cells]
            feature_arrays.append(arr)

        features_np = np.stack(feature_arrays, axis=0)
        return features_np

    def _close_files(self) -> None:
        """Close all open h5py file handles."""
        for i, f in enumerate(self._files):
            if f is not None:
                f.close()
                self._files[i] = None

    def _get_file(self, idx: int) -> h5py.File:
        """Get open h5py File object for the given index, opening it if not already open."""
        if self._files[idx] is None:
            with self._lock:
                if self._files[idx] is None:
                    self._files[idx] = h5py.File(self.hdf5_paths[idx], 'r')
        return self._files[idx]

    def __len__(self):
        """Return total number of cells across all .h5 files."""
        return len(self.cell_metadata.local_cell_idxs)

    def _load_cell_metadata(self) -> None:
        """Load metadata about each cell."""
        for path_idx, _ in enumerate(self.hdf5_paths):
            phase = self._get_file(path_idx)['Cells']['Phase']
            first_feature = self._get_features_np(
                path_idx, features=self.feature_names[0:1])[0]
            num_cells = first_feature.shape[1]  # already capped at num_cells

            # Per-cell values are stored (1, num_cells).
            if 'CellDeath' in phase:
                cell_deaths = self._get_features_np(path_idx,
                                                    features=['CellDeath'
                                                              ])[0][0]
            elif 'scenario' in self._get_file(path_idx)['Cells'].attrs:
                raise KeyError(f'{self.hdf5_paths[path_idx]}: synthetic file '
                               'has no Cells/Phase/CellDeath')
            else:  # real data without death labels: all censored
                cell_deaths = np.full(num_cells, np.nan)
            if self.ignores_death:
                cell_deaths = np.full(num_cells, np.nan)

            if 'First Frame' in phase and 'Last Frame' in phase:
                start, last = self._get_features_np(
                    path_idx, features=['First Frame', 'Last Frame'])[:, 0]
                start, last = start.astype(int), last.astype(int)
            else:  # not stored: first/last non-NaN frame of the first feature
                observed = ~np.isnan(
                    first_feature.T)  # (num_cells, num_frames)
                start = observed.argmax(axis=1)
                last = observed.shape[1] - 1 - observed[:, ::-1].argmax(axis=1)

            self.cell_metadata.file_idxs += [path_idx] * num_cells
            self.cell_metadata.local_cell_idxs += list(range(num_cells))
            self.cell_metadata.death_frames += list(cell_deaths)
            self.cell_metadata.start_frames += list(start)
            self.cell_metadata.end_frames += list(last)

        for idx in range(len(self)):
            landmark_frames = self._get_valid_landmarks(idx)
            self.cell_metadata.landmark_frames.append(landmark_frames)

        self.cell_metadata.landmarks_populated = True

    def compute_normalisation_stats(self):
        """Compute mean and std."""
        all_data = [
            cache.reshape(cache.shape[0], -1) for cache in self._features_cache
            if cache is not None
        ]
        all_data = np.concatenate(all_data, axis=1)

        means = np.nanmean(all_data, axis=1)
        stds = np.nanstd(all_data, axis=1)
        return means, stds

    def apply_normalisation(self):
        """Normalise features in the cache using stored means and stds."""
        safe_stds = np.where(self.stds < 1e-8, 1.0, self.stds)
        for file_idx, _ in enumerate(self.hdf5_paths):
            self._features_cache[file_idx] = (
                self._features_cache[file_idx] -
                self.means[:, np.newaxis,
                           np.newaxis]) / safe_stds[:, np.newaxis, np.newaxis]

    def get_normalisation_stats(self):
        """Get or compute normalisation stats."""
        if self.means is None or self.stds is None:
            self.means, self.stds = self.compute_normalisation_stats()
        return self.means, self.stds

    def _remove_invalid_samples(self):
        """Remove samples with no possible landmark frames."""
        valid_mask = [len(lf) > 0 for lf in self.cell_metadata.landmark_frames]
        self.cell_metadata.apply_mask(valid_mask)

    def _get_valid_landmarks(self, idx: int) -> np.ndarray:
        cell_metadata = self.cell_metadata.get_cell(idx)
        return self.landmarks_rule(
            cell_metadata.start_frames,
            cell_metadata.end_frames,
            cell_metadata.death_frames,
            self.min_length,
            self.horizon,
        )

    @property
    @abstractmethod
    def horizon(self) -> int | None:
        ...

    @property
    def ignores_death(self) -> bool:
        """True if the task's label doesn't involve death, so cells are
        treated as never dying: every cell is followed to its last frame,
        with no selection on survival."""
        return False

    @staticmethod
    @abstractmethod
    def landmarks_rule(start_frame: int, last_frame: int, death_frame: float,
                       min_len: int, horizon: int) -> np.ndarray:
        ...

    def cell_history(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Whole stored history of cell ``idx``: features (num_frames,
        num_features) with NaN replaced by 0, and the observed mask
        (num_frames,), False where any feature is NaN."""
        meta = self.cell_metadata
        features = self._features_cache[meta.file_idxs[idx]][
            ..., meta.local_cell_idxs[idx]].T
        nan = np.isnan(features)
        return np.where(nan, 0.0, features), ~nan.any(axis=1)

    def valid_landmarks(self, idx: int) -> np.ndarray:
        """Valid landmark frames of cell ``idx`` (the task's rule)."""
        return np.asarray(self.cell_metadata.landmark_frames[idx], dtype=int)

    def landmark_labels(self, idx: int,
                        landmarks: np.ndarray) -> dict[str, np.ndarray | None]:
        """Labels for cell ``idx`` at each of ``landmarks``: a dict of arrays
        with landmarks on axis 0 (None where unavailable, e.g. true values
        on real data). Implemented per task."""
        raise NotImplementedError(
            f'{type(self).__name__} does not define landmark labels')

    def _cell_pmf(self, idx: int) -> np.ndarray | None:
        """True per-frame event PMF of cell ``idx`` (synthetic data only)."""
        meta = self.cell_metadata
        pmfs = self._pmfs_cache[meta.file_idxs[idx]]
        return None if pmfs is None else pmfs[:, meta.local_cell_idxs[idx]]

    def __getitem__(self, idx: int):
        """Get a single sample from the dataset."""
        cell_metadata = self.cell_metadata.get_cell(idx)
        features = self._features_cache[cell_metadata.file_idxs][
            ..., cell_metadata.local_cell_idxs]

        event_indicator = 0 if np.isnan(cell_metadata.death_frames) else 1
        valid_landmark_fames = cell_metadata.landmark_frames

        landmark_frame = np.random.choice(valid_landmark_fames)

        features = features.T  # shape: (num_frames, num_features)

        return features, landmark_frame, event_indicator, cell_metadata

    def _load_true_variances(self) -> None:
        return

    @cached_property
    @abstractmethod
    def unobserved_variance(self) -> float | list[float] | None:
        ...

    def _get_unobserved_variances(self) -> Variances:
        file_path = self.hdf5_paths[0]
        scenario = load_scenario(file_path)
        if scenario is None:
            return None
        v = estimate_unobserved_variance(
            scenario,
            self.horizon,
            self.min_length,
            self.landmarks_rule,
            hazard_bins=self.bins,
        )
        return v

    @cached_property
    @abstractmethod
    def bins(self) -> np.ndarray | None:
        ...

    @property
    def context_feature_names(self) -> list[str]:
        """Column order of ``raw_context``: the features, plus the hazard
        (needed to roll the graph forward from the context)."""
        return list(self.feature_names) + ['Hazard']

    @cached_property
    def has_clean(self) -> bool:
        """Whether every file holds clean features and pre-frame-0 context
        (synthetic data only)."""
        for i in range(len(self.hdf5_paths)):
            cells = self._get_file(i)['Cells']
            if 'Clean' not in cells or 'Context' not in cells:
                return False
        return True

    def raw_context(self, idx: int) -> np.ndarray:
        """Clean history before frame 0 of cell ``idx``, (num_context, F + 1),
        columns ``context_feature_names`` (synthetic data only)."""
        meta = self.cell_metadata
        file_idx, cell = meta.file_idxs[idx], meta.local_cell_idxs[idx]
        if self._raw_context_cache[file_idx] is None:
            cells = self._get_file(file_idx)['Cells']
            if 'Context' not in cells:
                raise ValueError(
                    f'{self.hdf5_paths[file_idx]} has no context values')
            arr = np.stack([
                cells['Context'][n][:] for n in self.context_feature_names
            ])  # [F + 1, C, num_cells]
            if self.num_cells is not None:
                arr = arr[..., :self.num_cells]
            self._raw_context_cache[file_idx] = arr
        return self._raw_context_cache[file_idx][..., cell].T

    def raw_features(self, idx: int, clean: bool = False) -> np.ndarray:
        """Stored history of cell ``idx`` in raw units, (num_frames, F), NaN where
        unobserved. ``clean``: the noise-free signals (synthetic data only)."""
        meta = self.cell_metadata
        file_idx, cell = meta.file_idxs[idx], meta.local_cell_idxs[idx]
        if clean:
            cache = self._get_clean_cache(file_idx)
            return cache[..., cell].T
        x = self._features_cache[file_idx][..., cell].T
        if self.means is not None:
            x = x * self.stds + self.means
        return x

    def _get_clean_cache(self, file_idx: int) -> np.ndarray:
        """Clean features of one file, [F, T, C], loaded on first use."""
        if self._clean_features_cache[file_idx] is None:
            cells = self._get_file(file_idx)['Cells']
            if 'Clean' not in cells:
                raise ValueError(
                    f'{self.hdf5_paths[file_idx]} has no clean features')
            arr = np.stack([cells['Clean'][n][:] for n in self.feature_names])
            if self.num_cells is not None:
                arr = arr[..., :self.num_cells]
            self._clean_features_cache[file_idx] = arr
        return self._clean_features_cache[file_idx]


def collate_fn(batch: list[CellSample],
               device: str = 'cpu',
               pad_at: Literal['start', 'end'] = 'end') -> dict:
    """Collate function to pad sequences in batch to same length"""
    batch_dict = {}

    # max_seq_len = int(max([getattr(s, 'length') for s in batch]))
    max_seq_len = int(max([s.length for s in batch]))
    for f in fields(batch[0]):
        values = [getattr(s, f.name) for s in batch]

        if f.name == 'features':

            batch_dict[f.name] = _crop_and_pad(values, batch, pad_at,
                                               max_seq_len,
                                               device).to(torch.float32)
        elif f.name in ('mask', 'dense_mask'):
            batch_dict[f.name] = _crop_and_pad(values, batch, pad_at,
                                               max_seq_len,
                                               device).to(torch.int8)
        elif f.name.startswith('dense_'):  # per-frame dense labels
            batch_dict[f.name] = None if values[0] is None else _crop_and_pad(
                values, batch, pad_at, max_seq_len, device).to(torch.float32)
        elif f.name in [
                'time_to_event_bin',
                'event',
                'length',
                'event_indicator',
                'target_class',
        ]:
            batch_dict[f.name] = torch.tensor(values, device=device)

        elif f.name == 'hdf5_path':
            batch_dict[f.name] = values

        elif values[0] is None:
            batch_dict[f.name] = None

        else:
            batch_dict[f.name] = values

    return batch_dict


def _crop_and_pad(values: list[np.ndarray], samples_list: list[CellSample],
                  pad_at: Literal['start', 'end'], max_seq_len: int,
                  device: str) -> torch.Tensor:
    """Crop all features to (start_frame: landmark_framse) then pad all features in batch with zeros to same length"""
    # log.info(f'Padding dataset values, shape: {values[0].shape}')
    has_features_dim = values[0].ndim > 1
    if has_features_dim:
        num_features = values[0].shape[1]
        values_padded = torch.zeros(len(values),
                                    max_seq_len,
                                    num_features,
                                    device=device)
    else:
        values_padded = torch.zeros(len(values), max_seq_len, device=device)

    for idx, (sample_values, sample) in enumerate(zip(values, samples_list)):
        sample_values = sample_values[sample.start_frame:sample.landmark_frame]
        sample_len = sample_values.shape[0]
        if pad_at == 'start':
            values_padded[idx, -sample_len:] = torch.tensor(sample_values,
                                                            device=device)
        elif pad_at == 'end':
            values_padded[idx, :sample_len] = torch.tensor(sample_values,
                                                           device=device)

    return values_padded
