from __future__ import annotations
from pathlib import Path
import threading
from dataclasses import dataclass, field, fields
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal

import torch
import h5py
import numpy as np
from tqdm import tqdm

from PhagoPred.utils.logger import get_logger

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
            setattr(cell, f.name,
                    np.array(getattr(self, f.name), dtype=object)[idx])
        return cell

    def apply_mask(self, mask: np.ndarray) -> None:
        """Apply boolean mask to all metadata fields."""
        for f in fields(self):
            if not isinstance(getattr(self, f.name), list):
                continue
            setattr(self, f.name,
                    list(np.array(getattr(self, f.name), dtype=object)[mask]))


class CellDataset(torch.utils.data.Dataset, ABC):
    """Abstract base class for pytorch datasets using .h5 files."""

    def __init__(
        self,
        hdf5_paths: list[Path],
        feature_names: list[str],
        min_length: int = 50,
        means: np.ndarray | None = None,
        stds: np.ndarray | None = None,
        model_type: Literal['deep', 'classical'] = 'deep',
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
        self.hdf5_paths = hdf5_paths
        self.feature_names = feature_names
        self.min_length = min_length
        self.means = means
        self.stds = stds
        self.model_type = model_type

        self._lock = threading.Lock()
        self._files = [None] * len(hdf5_paths)

        self.full_len_frames = 0

        self._features_cache = [None] * len(hdf5_paths)
        self._pmfs_cache = [None] * len(hdf5_paths)

        self.cell_metadata = CellDatasetMetadata()

        self._load_features_cache()
        self._load_cell_metadata()
        self._remove_invalid_samples()

    def _load_features_cache(self) -> None:
        """Load features from all .h5 files into memory."""
        for i, _ in enumerate(self.hdf5_paths):
            features_cache = self._get_features_np(i)
            self._features_cache[
                i] = features_cache  # list[(num_frames, num_cells)]

            self.full_len_frames = max(self.full_len_frames,
                                       features_cache.shape[1])

            if 'PMFs' in self._get_file(i)['Cells']['Phase']:
                pmfs_cache = self._get_features_np(i, features=['PMFs'])[0]
                self._pmfs_cache[
                    i] = pmfs_cache  # list[(num_frames, num_cells)]

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
            feature_arrays.append(arr)

        features_np = np.stack(feature_arrays, axis=0)
        return features_np

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
            cell_deaths = self._get_features_np(
                path_idx, features=['CellDeath'])[0][0]  # shape = (num_cells)
            num_cells = len(cell_deaths)
            local_cell_idxs = np.arange(num_cells).astype(int)

            self.cell_metadata.file_idxs += [path_idx] * num_cells
            self.cell_metadata.local_cell_idxs += list(local_cell_idxs)
            self.cell_metadata.death_frames += list(cell_deaths)

            # Use first feature in list to determine start/end frames
            test_feature_data = self._get_features_np(
                path_idx, features=self.feature_names[0:1])
            test_feature_data = test_feature_data[0]
            area_data = test_feature_data.T  # shape: (num_cells, num_frames)

            not_nan_mask = ~np.isnan(area_data)
            start = not_nan_mask.argmax(axis=1)
            reversed_mask = not_nan_mask[:, ::-1]
            last = area_data.shape[1] - 1 - reversed_mask.argmax(axis=1)

            self.cell_metadata.start_frames += list(start + 1)
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
        for file_idx, _ in enumerate(self.hdf5_paths):
            self._features_cache[file_idx] = (
                self._features_cache[file_idx] -
                self.means[:, np.newaxis,
                           np.newaxis]) / self.stds[:, np.newaxis, np.newaxis]

    def get_normalisation_stats(self):
        """Get or compute normalisation stats."""
        if self.means is None or self.stds is None:
            self.means, self.stds = self.compute_normalisation_stats()
        return self.means, self.stds

    def _remove_invalid_samples(self):
        """Remove samples with no possible landmark frames."""
        valid_mask = [len(lf) > 0 for lf in self.cell_metadata.landmark_frames]
        self.cell_metadata.apply_mask(valid_mask)

    @abstractmethod
    def _get_valid_landmarks(self, idx: int) -> list:
        """Get valid lanndmark frames for the given cell index."""

    def __getitem__(self, idx: int):
        """Get a single sample from the dataset."""
        cell_metadata = self.cell_metadata.get_cell(idx)
        features = self._features_cache[cell_metadata.file_idxs][
            ..., cell_metadata.local_cell_idxs]

        start_frame = cell_metadata.start_frames
        last_frame = cell_metadata.end_frames if np.isnan(
            cell_metadata.death_frames) else cell_metadata.death_frames
        event_indicator = 0 if np.isnan(cell_metadata.death_frames) else 1
        valid_landmark_fames = cell_metadata.landmark_frames

        landmark_frame = np.random.choice(valid_landmark_fames)

        if self.model_type == 'deep':
            features = features[:,
                                int(landmark_frame - start_frame):
                                int(last_frame + 1
                                    )]  # For deep models, crop to known frames
        features = features.T  # shape: (num_frames, num_features)

        return features, landmark_frame, event_indicator, cell_metadata


def collate_fn(batch: list[CellSample],
               device: str = 'cpu',
               pad_at: Literal['start', 'end'] = 'end') -> dict:
    """Collate function to pad sequences in batch to same length"""
    batch_dict = {}

    max_seq_len = int(max([getattr(s, 'length') for s in batch]))
    for f in fields(batch[0]):
        values = [getattr(s, f.name) for s in batch]

        if f.name == 'features':
            batch_dict[f.name] = _pad_values(values, pad_at, max_seq_len,
                                             device).to(torch.float32)
        elif f.name == 'mask':
            batch_dict[f.name] = _pad_values(values, pad_at, max_seq_len,
                                             device).to(torch.int8)
        elif f.name in [
                'time_to_event_bin',
                'event',
                'length',
                'event_indicator',
        ]:
            batch_dict[f.name] = torch.tensor(values, device=device)

        elif f.name == 'hdf5_path':
            batch_dict[f.name] = values

        elif values[0] is None:
            batch_dict[f.name] = None

        else:
            batch_dict[f.name] = values

    return batch_dict


def _pad_values(values: list[np.ndarray], pad_at: Literal['start', 'end'],
                max_seq_len: int, device: str) -> torch.Tensor:
    """Pad all features in batch with zeros to same length"""
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
    for idx, sample in enumerate(values):
        sample_len = sample.shape[0]
        if pad_at == 'start':
            values_padded[idx, -sample_len:] = torch.tensor(sample,
                                                            device=device)
        elif pad_at == 'end':
            values_padded[idx, :sample_len] = torch.tensor(sample,
                                                           device=device)
    return values_padded
