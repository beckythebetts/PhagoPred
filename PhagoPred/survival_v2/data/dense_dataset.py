"""Dense-supervision view over a task dataset (binary or survival).

One sample per cell covering its whole observable history, with every
per-landmark label laid out per frame plus a validity mask, so a causal model
can be trained on every valid landmark in a single forward pass.

The task dataset defines the labels (``landmark_labels``) and which landmarks
are valid; this view only arranges them. A label ``x`` becomes the per-frame
array ``dense_x``. Frame convention matches the sampled view: a prediction at
landmark L uses features [start, L), so per-frame arrays are indexed by the
last input frame f = L - 1 and, after _crop_and_pad, line up with the model's
per-timestep outputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import torch

from .dataset import (
    CellDataset,
    CellSample,
    collate_fn,
    ArrayType_T,
    ScalarType_T,
    PathType_T,
    ListType_T,
)
from .binary_dataset import BinaryCellDataset
from .survival_dataset import SurvivalCellDataset


@dataclass
class DenseCell(CellSample[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """A whole cell with per-frame labels (one sample per cell)."""
    dense_mask: ArrayType_T  # (num_frames,) True where f + 1 is a valid landmark


@dataclass
class DenseBinaryCell(DenseCell[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """Per-frame BinaryCellDataset.landmark_labels."""
    dense_event: ArrayType_T  # (num_frames,)
    dense_event_probability: ArrayType_T | None = None  # (num_frames,)


@dataclass
class DenseSurvivalCell(DenseCell[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """Per-frame SurvivalCellDataset.landmark_labels."""
    dense_time_to_event: ArrayType_T  # (num_frames,)
    dense_event_indicator: ArrayType_T  # (num_frames,)
    dense_time_to_event_bin: ArrayType_T | None = None  # (num_frames,)
    dense_binned_pmf: ArrayType_T | None = None  # (num_frames, num_bins)


DenseBinaryCellSample = DenseBinaryCell[np.ndarray, float, Path, list]
DenseBinaryCellBatch = DenseBinaryCell[Union[torch.Tensor, np.ndarray],
                                       Union[torch.Tensor, np.ndarray],
                                       list[Path], list[list]]
DenseSurvivalCellSample = DenseSurvivalCell[np.ndarray, float, Path, list]
DenseSurvivalCellBatch = DenseSurvivalCell[Union[torch.Tensor, np.ndarray],
                                           Union[torch.Tensor, np.ndarray],
                                           list[Path], list[list]]

_SAMPLE_TYPES: dict[type, type] = {
    BinaryCellDataset: DenseBinaryCell,
    SurvivalCellDataset: DenseSurvivalCell,
}


class DenseCellDataset(torch.utils.data.Dataset):
    """View over a task dataset returning one dense sample per cell.

    Shares the wrapped dataset's (already normalised) feature cache and
    metadata, so nothing is reloaded; anything not defined here (e.g.
    feature_names, means, prediction_horizon, event_time_bins,
    get_pos_weight) is read from the wrapped dataset.
    """

    def __init__(self, dataset: CellDataset):
        sample_type = next((sample for task, sample in _SAMPLE_TYPES.items()
                            if isinstance(dataset, task)), None)
        if sample_type is None:
            raise TypeError(
                f'No dense view for {type(dataset).__name__}; supported: '
                f'{[t.__name__ for t in _SAMPLE_TYPES]}')
        self.dataset = dataset
        self.sample_type = sample_type

    def __getattr__(self, name: str):
        # Only called for attributes not found normally; guard against
        # recursion before self.dataset is set (e.g. while unpickling).
        if name == 'dataset':
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> DenseCell:
        ds = self.dataset
        meta = ds.cell_metadata
        features, observed = ds.cell_history(idx)
        landmarks = ds.valid_landmarks(idx)
        num_frames = features.shape[0]
        rows = landmarks - 1  # last input frame of each landmark

        dense = {'dense_mask': np.zeros(num_frames, dtype=bool)}
        dense['dense_mask'][rows] = True
        for name, values in ds.landmark_labels(idx, landmarks).items():
            if values is None:
                dense[f'dense_{name}'] = None
                continue
            values = np.asarray(values)
            per_frame = np.zeros((num_frames, ) + values.shape[1:],
                                 dtype=np.float32)
            per_frame[rows] = values
            dense[f'dense_{name}'] = per_frame

        start_frame = int(meta.start_frames[idx])
        # _crop_and_pad keeps [start_frame, landmark_frame), i.e. the inputs
        # for every landmark up to and including the last one.
        last_landmark = int(landmarks.max())
        return self.sample_type(
            features=features,
            mask=observed,
            length=last_landmark - start_frame,
            cell_idx=meta.local_cell_idxs[idx],
            hdf5_path=ds.hdf5_paths[meta.file_idxs[idx]],
            start_frame=start_frame,
            landmark_frame=last_landmark,
            last_frame=meta.end_frames[idx],
            **dense,
        )

    def make_sampler(self) -> torch.utils.data.RandomSampler:
        """Each cell once per epoch; landmark weighting comes from the loss."""
        return torch.utils.data.RandomSampler(self)


def dense_collate_fn(batch: list[DenseCell],
                     device: str = 'cpu',
                     pad_at: Literal['start', 'end'] = 'end') -> DenseCell:
    """Pad a batch of dense samples (binary or survival) to the same length."""
    batch_dict = collate_fn(batch, device, pad_at)
    return type(batch[0])(**batch_dict)
