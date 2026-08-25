from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
from pathlib import Path

import numpy as np
import torch

from .dataset import (
    CellSample,
    CellDataset,
    CellDatasetMetadata,
    collate_fn,
    ArrayType_T,
    ScalarType_T,
    PathType_T,
    ListType_T,
)


@dataclass
class BinaryClassCell(CellSample[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """Data class for binary classification samples."""
    target_class: ScalarType_T


BinaryClassSample = BinaryClassCell[
    np.ndarray,
    float,
    Path,
    list,
]

BinaryClassBatch = BinaryClassCell[
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    list[Path],
    list[list],
]


class BinaryClassDataset(CellDataset):

    def __init__(self, *args, class_dict: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_dict = {Path(p): c for p, c in class_dict.items()}

    def _get_valid_landmarks(self, idx):
        cell_metadata = self.cell_metadata.get_cell(idx)
        last_frame = cell_metadata.end_frames
        return [last_frame]

    def __getitem__(self, idx: int) -> BinaryClassSample:
        features, landmark_frame, event_indicator, cell_metadata = super(
        ).__getitem__(idx)
        nan_mask = np.isnan(features).any(axis=1)
        features = np.where(np.isnan(features), 0.0, features)

        target_class = self.class_dict[self.hdf5_paths[
            cell_metadata.file_idxs]]

        return BinaryClassSample(
            features=features,
            mask=~nan_mask,
            length=landmark_frame - cell_metadata.start_frames + 1,
            cell_idx=cell_metadata.local_cell_idxs,
            hdf5_path=self.hdf5_paths[cell_metadata.file_idxs],
            start_frame=cell_metadata.start_frames,
            landmark_frame=landmark_frame,
            last_frame=cell_metadata.end_frames,
            target_class=target_class)


def binary_class_collate_fn(
        batch: list[BinaryClassSample],
        device: str = 'cpu',
        pad_at: Literal['start', 'end'] = 'end') -> BinaryClassBatch:
    return BinaryClassBatch(**collate_fn(batch, device, pad_at))
