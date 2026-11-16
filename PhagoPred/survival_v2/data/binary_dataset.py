from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
from pathlib import Path
from functools import cached_property

import numpy as np
import torch
import h5py
from tqdm import tqdm

from .dataset import (
    CellSample,
    CellDataset,
    collate_fn,
    ArrayType_T,
    ScalarType_T,
    PathType_T,
    ListType_T,
)


@dataclass
class BinaryCell(CellSample[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """Data class for binary classification samples."""
    event: ScalarType_T
    event_probability: ScalarType_T | None


BinaryCellSample = BinaryCell[
    np.ndarray,
    float,
    Path,
    list,
]

BinaryCellBatch = BinaryCell[
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    list[Path],
    list[list],
]


class BinaryCellDataset(CellDataset):
    """Pytorch dataset for binary classification, returning BinaryCellSample objects."""

    def __init__(self, *args, prediction_horizon: int, **kwargs):
        self.prediction_horizon = prediction_horizon
        super().__init__(*args, **kwargs)

    @property
    def horizon(self):
        return self.prediction_horizon

    @staticmethod
    def landmarks_rule(
        start_frame: int,
        last_frame: int,
        death_frame: float,
        min_len: int,
        horizon: int,
    ) -> np.ndarray:
        if np.isnan(death_frame):
            last_landmark = last_frame - horizon
        else:
            last_landmark = death_frame
        return np.arange(int(start_frame + min_len), int(last_landmark))

    def landmark_labels(self, idx: int,
                        landmarks: np.ndarray) -> dict[str, np.ndarray | None]:
        """Labels at each landmark L (prediction from features [start, L)):

        * event: death in [L + 1, L + H] (0 if censored)
        * event_probability: true P(death in [L + 1, L + H] | alive at L),
          from the stored PMF; None without one (real data)
        """
        landmarks = np.asarray(landmarks, dtype=int)
        time_to_event = float(self.cell_metadata.death_frames[idx]) - landmarks
        # NaN (censored) compares False, so censored cells get 0.
        event = ((time_to_event >= 1) &
                 (time_to_event <= self.prediction_horizon)).astype(np.float32)

        event_probability = None
        pmf = self._cell_pmf(idx)
        if pmf is not None:
            cif = np.cumsum(pmf, dtype=np.float64)
            survival = np.clip(1.0 - cif[landmarks], 0.0, 1.0)
            horizon_end = np.minimum(landmarks + self.prediction_horizon,
                                     len(cif) - 1)
            in_horizon = (cif[horizon_end] - cif[landmarks]).astype(float)
            event_probability = np.clip(
                np.divide(in_horizon,
                          survival,
                          out=np.zeros_like(in_horizon),
                          where=survival > 0.0), 0.0, 1.0).astype(np.float32)
        return {'event': event, 'event_probability': event_probability}

    def __getitem__(self, idx: int) -> BinaryCellSample:
        """One sample: cell ``idx`` at a random valid landmark."""
        meta = self.cell_metadata
        features, observed = self.cell_history(idx)
        landmark = int(np.random.choice(self.valid_landmarks(idx)))
        labels = self.landmark_labels(idx, [landmark])
        prob = labels['event_probability']
        start_frame = meta.start_frames[idx]

        return BinaryCellSample(
            features=features,
            mask=observed,
            length=landmark - start_frame,
            cell_idx=meta.local_cell_idxs[idx],
            hdf5_path=self.hdf5_paths[meta.file_idxs[idx]],
            start_frame=start_frame,
            landmark_frame=landmark,
            last_frame=meta.end_frames[idx],
            event=int(labels['event'][0]),
            event_probability=None if prob is None else float(prob[0]),
        )

    def get_pos_weight(self) -> float:
        """Num -ve / num +ve over every valid (cell, landmark) pair, for the
        weighted BCE loss. Exact, and the same for sampled and dense
        training, which both cover every pair equally."""
        events = np.concatenate([
            self.landmark_labels(idx, self.valid_landmarks(idx))['event']
            for idx in range(len(self))
        ])
        n_pos = float(events.sum())
        if n_pos == 0:
            return 1.0
        return (len(events) - n_pos) / n_pos

    @cached_property
    def unobserved_variance(self) -> float | None:
        v = self._get_unobserved_variances()
        if v is None:
            return None
        else:
            return v.event_probability

    @cached_property
    def total_variance(self) -> float | None:
        if any(pmfs is None for pmfs in self._pmfs_cache):
            return None
        probs = [
            self.landmark_labels(
                idx, self.valid_landmarks(idx))['event_probability']
            for idx in range(len(self))
        ]
        return float(np.var(np.concatenate(probs)))

    @cached_property
    def bins(self):
        return None


def binary_collate_fn(batch: list[CellSample],
                      device: str = 'cpu',
                      pad_at: Literal['start', 'end'] = 'end') -> dict:
    """Collate function to pad sequences in batch to same length"""
    batch_dict = collate_fn(batch, device, pad_at)
    return BinaryCellSample(**batch_dict)
