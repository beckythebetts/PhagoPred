from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
from pathlib import Path
from functools import cached_property

import numpy as np
import torch
import h5py
from tqdm import tqdm

from .graph_synthetic.dataset_file import read_scenario_dict
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
    """Pytorch dataset for prediction over a horizon after each landmark,
    returning BinaryCellSample objects.

    ``target`` sets the label:

    * 'event': death in the horizon (0/1), a binary classification target.
    * 'hazard': the mean value of the Hazard node over the horizon frames,
      times the scenario's calibrated ``scale`` (synthetic data only), a
      regression target. It is the latent node behind the event label, read
      off without the death process: noiseless, and on the same scale in
      every scenario (calibration sets scale * Hazard's SD). The death-rate
      offset ``delta`` is left out: it only sets how many cells die.
    """

    def __init__(self,
                 *args,
                 prediction_horizon: int,
                 target: Literal['event', 'hazard'] = 'event',
                 **kwargs):
        if target not in ('event', 'hazard'):
            raise ValueError(f'unknown target {target!r}')
        self.prediction_horizon = prediction_horizon
        self.target = target
        self._hazard_cache: dict[int, np.ndarray] = {}
        super().__init__(*args, **kwargs)

    @property
    def horizon(self):
        return self.prediction_horizon

    @property
    def is_regression(self) -> bool:
        return self.target == 'hazard'

    def _hazard_values(self, file_idx: int) -> np.ndarray:
        """scale * Hazard for every frame and cell of one file,
        [num_frames, num_cells]."""
        if file_idx not in self._hazard_cache:
            path = self.hdf5_paths[file_idx]
            scenario = read_scenario_dict(path)
            if scenario is None or scenario.get('hazard_func') is None:
                raise ValueError(f"{path}: target='hazard' needs a synthetic "
                                 'dataset with a calibrated hazard function')
            func = scenario['hazard_func']
            if func['kind'] != 'offset_sigmoid':
                raise ValueError(f'{path}: unsupported hazard function '
                                 f"{func['kind']!r}")
            with h5py.File(path, 'r') as f:
                z = f['Cells']['Hazard'][:]
            if self.num_cells is not None:
                z = z[:, :self.num_cells]
            self._hazard_cache[file_idx] = (
                func['params'].get('scale', 1.0) * z)
        return self._hazard_cache[file_idx]

    @staticmethod
    def landmarks_rule(
        start_frame: int,
        last_frame: int,
        death_frame: float,
        min_len: int,
        horizon: int,
    ) -> np.ndarray:
        # Every landmark needs its whole horizon observed, dying cell or not.
        # Letting dying cells keep landmarks up to death made the last
        # `horizon` frames all-event (only cells about to die have landmarks
        # there): label rate 1.0 vs true probability ~0.26.
        last_landmark = last_frame - horizon
        if not np.isnan(death_frame):
            last_landmark = min(last_landmark, death_frame)
        return np.arange(int(start_frame + min_len), int(last_landmark))

    def landmark_labels(self, idx: int,
                        landmarks: np.ndarray) -> dict[str, np.ndarray | None]:
        """Labels at each landmark L (prediction from features [start, L)):

        * event: death in [L + 1, L + H] (0 if censored)
        * event_probability: true P(death in [L + 1, L + H] | alive at L),
          from the stored PMF; None without one (real data)
        """
        landmarks = np.asarray(landmarks, dtype=int)
        if self.is_regression:
            # Mean (scaled) Hazard over frames L + 1 .. L + H, the frames
            # whose rates make up the event label. Deterministic given the window,
            # so it is also its own true value.
            meta = self.cell_metadata
            z = self._hazard_values(meta.file_idxs[idx])[:,
                                                         meta.local_cell_idxs[idx]]
            frames = landmarks[:, None] + np.arange(1,
                                                    self.prediction_horizon + 1)
            value = z[frames].mean(axis=1).astype(np.float32)
            return {'event': value, 'event_probability': value}
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
            event=(float(labels['event'][0])
                   if self.is_regression else int(labels['event'][0])),
            event_probability=None if prob is None else float(prob[0]),
        )

    def get_pos_weight(self) -> float:
        """Num -ve / num +ve over every valid (cell, landmark) pair, for the
        weighted BCE loss. Exact, and the same for sampled and dense
        training, which both cover every pair equally."""
        if self.is_regression:
            return 1.0  # no classes to balance
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
        if self.is_regression:
            # The hazard window is fully determined by the observed window.
            return 0.0
        v = self._get_unobserved_variances()
        if v is None:
            return None
        else:
            return v.event_probability

    @cached_property
    def total_variance(self) -> float | None:
        if not self.is_regression and any(pmfs is None
                                          for pmfs in self._pmfs_cache):
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
