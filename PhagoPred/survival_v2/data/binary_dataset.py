from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Union
from pathlib import Path

import numpy as np
import torch
import h5py
from tqdm import tqdm

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
from .graph_synthetic.scenarios import ALL_CFGS
from .graph_synthetic.generate_datasets import estimate_variances


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

    def _get_valid_landmarks(self, idx: int) -> list:
        """Get valid landmark frames.
        * Sequence length must be >= self.min_length
        * Landmark_frame + prediction_horizon must be <= censoring frame.
        """
        cell_metadata = self.cell_metadata.get_cell(idx)
        start_frame = cell_metadata.start_frames
        last_frame = cell_metadata.end_frames if np.isnan(
            cell_metadata.death_frames) else cell_metadata.death_frames
        event_indicator = 0 if np.isnan(cell_metadata.death_frames) else 1
        min_landmark_dist = self.min_length
        if event_indicator == 0:
            max_landmark_dist = last_frame - start_frame - self.prediction_horizon
        else:
            max_landmark_dist = last_frame - start_frame

        if min_landmark_dist <= 0:
            return []
        if max_landmark_dist - min_landmark_dist <= 0:
            return []

        valid_landmarks = list(
            range(int(start_frame + min_landmark_dist),
                  int(start_frame + max_landmark_dist)))
        return valid_landmarks

    def __getitem__(self, idx: int) -> BinaryCellSample:
        """Get a single sample as BinaryCellSample dataclass."""
        features, landmark_frame, event_indicator, cell_metadata = super(
        ).__getitem__(idx)
        last_frame = cell_metadata.end_frames if np.isnan(
            cell_metadata.death_frames) else cell_metadata.death_frames
        time_to_event = last_frame - landmark_frame

        event = int(time_to_event <= self.prediction_horizon
                    and event_indicator == 1)
        # features[
        #     landmark_frame:] = np.nan  # remove features after landmark frame

        event_probability = self._get_item_event_probability(
            cell_metadata, landmark_frame)

        nan_mask = np.isnan(features).any(axis=1)
        features = np.where(np.isnan(features), 0.0, features)

        return BinaryCellSample(
            features=features,
            mask=~nan_mask,
            length=landmark_frame - cell_metadata.start_frames,
            cell_idx=cell_metadata.local_cell_idxs,
            hdf5_path=self.hdf5_paths[cell_metadata.file_idxs],
            start_frame=cell_metadata.start_frames,
            landmark_frame=landmark_frame,
            last_frame=cell_metadata.end_frames,
            event=event,
            event_probability=event_probability,
        )

    def _get_item_event_probability(self, cell_metadata: CellDatasetMetadata,
                                    landmark_frame: int) -> float:
        """Get proabability of event occuring for sample for which underlying PMF is known.
        Returns
        -------
            event_probability [float]
        """
        event_probability = None
        if self._pmfs_cache[cell_metadata.file_idxs] is not None:
            full_pmf = self._pmfs_cache[
                cell_metadata.file_idxs][:, cell_metadata.local_cell_idxs]
            survival = 1.0 - full_pmf[:landmark_frame].sum()
            survival = np.clip(survival, 0.0, 1.0)
            horizon_end = min(landmark_frame + self.prediction_horizon + 1,
                              len(full_pmf))
            if survival > 0.0:
                event_probability = full_pmf[landmark_frame:horizon_end].sum(
                ) / survival
            else:
                event_probability = 0.0
            event_probability = float(np.clip(event_probability, 0.0, 1.0))
        return event_probability

    def get_pos_weight(self, n_samples: int = 1000) -> float:
        """Get positive weight for weighted BCE loss.
        Num -ve / num +ve, estimated from n_samples random samples.
        """
        indices = np.random.choice(len(self),
                                   size=min(n_samples, len(self)),
                                   replace=False)
        events = [self[int(i)].event for i in indices]
        n_pos = sum(events)
        n_neg = len(events) - n_pos
        if n_pos == 0:
            return 1.0
        return float(n_neg) / float(n_pos)

    def _load_true_variances(self) -> None:
        file_path = self.hdf5_paths[0]
        # Match on the '<cfg.filename>_{train,val,cal}' stem exactly, not a
        # substring: e.g. 'linear_chain_high_ar' is a substring of
        # 'nonlinear_chain_high_ar_train.h5' (via "non" + "linear..."), so a
        # plain `in` check would silently pick the wrong scenario's graph and
        # calibration.
        stem_prefix = Path(file_path).stem.rsplit('_', 1)[0]
        for cfg in ALL_CFGS:
            if cfg.filename == stem_prefix:
                cfg.load(Path(file_path).parent)

                variances = estimate_variances(
                    cfg.graph,
                    cfg.hazard_calibration_func,
                    self.prediction_horizon,
                    self.full_len_frames,
                    self.min_length,
                    1000,
                    1000,
                )
                # landmark_dist = self._sample_landmark_distribution(
                #     n_samples=5000)
                # variances = estimate_variances(
                #     cfg.graph,
                #     cfg.hazard_calibration_func,
                #     self.prediction_horizon,
                #     cfg.num_frames - self.prediction_horizon,
                #     base_sample_size=200,
                #     branch_sample_size=1000,
                #     warm_up_steps=self.min_length,
                #     landmark_distribution=landmark_dist,
                # )
                # cdf['Total'] is shape (prediction_horizon,);
                # index -1 gives cumulative P(death within full horizon)
                self.total_variances = float(variances['cdf']['Total'][-1])
                self.unobserved_variances = float(
                    variances['cdf']['Unobserved'][-1])
                return

    def _compute_variance(self, sample_size: int = 1000) -> None:
        vals = []
        for _ in tqdm(range(sample_size), 'Smapling cells for cdf variance'):
            cell_idx = np.random.randint(len(self.cell_metadata))
            cell = self[cell_idx]
            vals.append(cell.event_probability)

        return np.var(vals)


def binary_collate_fn(batch: list[CellSample],
                      device: str = 'cpu',
                      pad_at: Literal['start', 'end'] = 'end') -> dict:
    """Collate function to pad sequences in batch to same length"""
    batch_dict = collate_fn(batch, device, pad_at)
    return BinaryCellSample(**batch_dict)
