from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from .dataset import (CellSample, CellDataset, CellDatasetMetadata, collate_fn,
                      ArrayType_T, ScalarType_T, PathType_T, ListType_T)

from .graph_synthetic.scenarios import ScenarioCfg, ALL_CFGS
from .graph_synthetic.generate_datasets import estimate_variances

from PhagoPred.utils.logger import get_logger

logger = get_logger()


@dataclass
class SurvivalCell(CellSample[
        ArrayType_T,
        ScalarType_T,
        PathType_T,
        ListType_T,
]):
    """Data class for survival analysis samples."""
    time_to_event: ScalarType_T
    time_to_event_bin: ScalarType_T
    event_indicator: ScalarType_T
    pmf: ListType_T | None
    binned_pmf: ListType_T | None


SurvivalCellSample = SurvivalCell[
    np.ndarray,
    float,
    Path,
    list,
]
SurvivalCellBatch = SurvivalCell[
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    list[Path],
    list[list],
]


class SurvivalCellDataset(CellDataset):
    """Pytorch dataset for survival analysis, returning SurvivalCellSample objects."""

    def __init__(self,
                 *args,
                 num_bins: int,
                 max_time_to_death: int,
                 event_time_bins: np.ndarray | list | None = None,
                 **kwargs):

        self.num_bins = num_bins
        self.max_time_to_death = max_time_to_death
        self.event_time_bins = np.array(
            event_time_bins) if event_time_bins is not None else None

        super().__init__(*args, **kwargs)

        # if self.event_time_bins is None:
        #     self._compute_bins()

    def _compute_bins(self, num_samples: int = 10000) -> None:
        """Compute event time bins based on quantiles of observed event times in the dataset."""
        times = []
        events = 0
        for _ in range(num_samples):
            idx = np.random.randint(len(self))
            item: SurvivalCellSample = self[idx]

            e = item.event_indicator
            t = item.time_to_event

            events += e
            # Only within-horizon events shape the bin edges: deaths beyond
            # max_time_to_death become administratively censored, and including
            # their (large) times would push a quantile above bins[-1] =
            # max_time_to_death + 1, breaking monotonicity in np.digitize.
            if e == 1 and t <= self.max_time_to_death:
                times.append(t)

        times = np.array(times)
        bins = np.quantile(times, np.linspace(0, 1,
                                              self.num_bins + 1)).astype(int)
        # bins[-1] = bins[-1] * 1e5
        bins[-1] = self.max_time_to_death + 1
        self.event_time_bins = bins

    def get_bins(self) -> np.ndarray:
        """Return event time bins."""
        if self.event_time_bins is None:
            self._compute_bins()
        return self.event_time_bins

    def get_bin_weights(self, num_samples: int = 1000) -> list:
        """Inverse frequency of bins (for uncensored samles only), for weighting NLL loss."""
        counts = np.zeros(self.num_bins)
        for _ in range(num_samples):
            idx = np.random.randint(len(self))
            sample = self[idx]
            if sample.event_indicator == 1:
                counts[sample.time_to_event_bin] += 1
        counts = np.where(counts == 0, 1, counts)
        weights = counts.sum() / (self.num_bins * counts)
        return weights.tolist()

    def _get_valid_landmarks(self, idx: int) -> list:
        """Allowed landmark frames for cell {idx}.
        * Total length must be >= self.min_length.
        * ==Time to death must be <= self.max_time_to_death if cell is uncensored.==
        """

        cell_metadata = self.cell_metadata.get_cell(idx)
        last_frame = cell_metadata.end_frames if np.isnan(
            cell_metadata.death_frames) else cell_metadata.death_frames

        # event_indicator = 0 if np.isnan(cell_metadata.death_frames) else 1
        # if event_indicator == 0:
        min_landmark_dist = self.min_length
        # else:
        #     min_landmark_dist = max(
        #         self.min_length, last_frame - self.max_time_to_death -
        #         cell_metadata.start_frames)

        if last_frame <= cell_metadata.start_frames + min_landmark_dist:
            return []

        valid_landmarks = list(
            range(int(cell_metadata.start_frames + min_landmark_dist),
                  int(last_frame + 1)))
        return valid_landmarks

    def __getitem__(self, idx: int) -> SurvivalCellSample:
        """Get a single sample from the dataset as a SurvivalCellSample dataclass."""
        logger.debug(
            f'Getting item, event time abins are: {self.event_time_bins}')
        features, landmark_frame, event_indicator, cell_metadata = super(
        ).__getitem__(idx)

        last_frame = cell_metadata.end_frames if np.isnan(
            cell_metadata.death_frames) else cell_metadata.death_frames
        time_to_event = last_frame - landmark_frame

        time_to_event_bin = None
        if self.event_time_bins is not None:
            if time_to_event <= self.event_time_bins[-1]:
                time_to_event_bin = np.digitize(time_to_event,
                                                self.event_time_bins[1:])
                time_to_event_bin = min(time_to_event_bin, self.num_bins - 1)
            else:
                event_indicator = 0
                time_to_event_bin = self.num_bins

        pmf, binned_pmf = self._get_item_pmf(cell_metadata, landmark_frame)

        nan_mask = np.isnan(features).any(axis=1)
        features = np.where(np.isnan(features), 0.0, features)

        return SurvivalCellSample(
            features=features,
            mask=~nan_mask,
            length=landmark_frame - cell_metadata.start_frames + 1,
            cell_idx=cell_metadata.local_cell_idxs,
            hdf5_path=self.hdf5_paths[cell_metadata.file_idxs],
            start_frame=cell_metadata.start_frames,
            landmark_frame=landmark_frame,
            last_frame=cell_metadata.end_frames,
            time_to_event=time_to_event,
            time_to_event_bin=time_to_event_bin,
            event_indicator=event_indicator,
            pmf=pmf,
            binned_pmf=binned_pmf)

    def _get_item_pmf(self, cell_metadata: CellDatasetMetadata,
                      landmark_frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Get pmf and binned pmf for cell {cell_idx} if underlying PMF known
        Returns
        -------
            PMF
            Binned PMF
        """

        pmf = None
        binned_pmf = None
        if self._pmfs_cache[0] is not None and self.event_time_bins is not None:
            full_pmf = self._pmfs_cache[
                cell_metadata.file_idxs][:, cell_metadata.local_cell_idxs]
            survival = 1.0 - full_pmf[:landmark_frame].sum()
            survival = np.clip(survival, 0.0, 1.0)
            if survival > 0.0:
                pmf = full_pmf[landmark_frame:] / survival
            else:
                pmf = full_pmf[landmark_frame:] * 0.0

            pmf = np.clip(pmf, 0.0, 1.0)

            binned_pmf = np.array([
                pmf[int(self.event_time_bins[i]
                        ):int(self.event_time_bins[i + 1])].sum()
                for i in range(len(self.event_time_bins) - 1)
            ])

            binned_pmf = np.round(binned_pmf, decimals=4)
            binned_pmf = np.clip(binned_pmf, a_min=0.0, a_max=1.0)

            # s = binned_pmf.sum()
            # if s > 0:
            #     binned_pmf = binned_pmf / s

        return pmf, binned_pmf

    def _load_true_variances(self) -> None:
        file_path = self.hdf5_paths[0]
        for cfg in ALL_CFGS:
            if cfg.filename in Path(file_path).name:
                cfg.load(Path(file_path).parent)
                variances = estimate_variances(
                    cfg.graph,
                    cfg.hazard_calibration_func,
                    self.event_time_bins[-1],
                    self.full_len_frames,
                    self.min_length,
                    trunk_sample_size=1000,
                    branch_smaple_size=1000,
                    hazard_bins=self.event_time_bins,
                    max_time_to_death=self.max_time_to_death,
                )
                # landmark_dist = self._sample_landmark_distribution(
                #     n_samples=5000)
                # variances = estimate_variances(
                #     cfg.graph,
                #     cfg.hazard_calibration_func,
                #     self.event_time_bins[-1],
                #     cfg.num_frames - self.event_time_bins[-1],
                #     base_sample_size=200,
                #     branch_sample_size=1000,
                #     warm_up_steps=self.min_length,
                #     hazard_bins=self.event_time_bins,
                #     landmark_distribution=landmark_dist,
                # )
                self.total_variances = {}
                self.unobserved_variances = {}
                for t in ('hazard', 'pmf'):
                    self.total_variances[t] = variances[t]['Total']
                    self.unobserved_variances[t] = variances[t]['Unobserved']
                return

    # def _compute_variance(self, sample_size: int = 1000) -> np.ndarray:
    #     """Per-bin variance of the true discrete-time hazard across cells, under this
    #     dataset's own (per-cell, one random landmark) sampling.

    #     This is the hazard-form mean-estimator MSE — i.e. the total variance the model
    #     is benchmarked against — computed exactly from the cached PMFs (no branching MC).
    #     """
    #     vals = []
    #     for _ in tqdm(range(sample_size),
    #                   'Sampling cells for hazard variance'):
    #         cell_idx = np.random.randint(len(self))
    #         cell = self[cell_idx]
    #         pmf = cell.binned_pmf
    #         if pmf is None:
    #             continue
    #         pmf = np.asarray(pmf, dtype=float)

    #         # survival to the START of each bin: S_{i-1} = 1 - sum_{j<i} pmf_j
    #         survival_before = 1.0 - (np.cumsum(pmf) - pmf)  # exclusive cumsum
    #         # discrete-time hazard per bin: h_i = pmf_i / S_{i-1}
    #         hazard = np.divide(pmf,
    #                            survival_before,
    #                            out=np.zeros_like(pmf),
    #                            where=survival_before > 1e-12)
    #         vals.append(hazard)

    #     return np.var(vals, axis=0)


def survival_collate_fn(batch: list[CellSample],
                        device: str = 'cpu',
                        pad_at: Literal['start', 'end'] = 'end') -> dict:
    """Collate function to pad sequences in batch to same length"""
    batch_dict = collate_fn(batch, device, pad_at)
    return SurvivalCellSample(**batch_dict)


def plot_samples_per_bin(dataset, num_samples=10000):
    """
    Plot number of samples per event-time bin.

    Args:
        dataset (CellDataset): dataset with event_time_bins defined
        num_samples (int): number of random samples to draw
        show (bool): whether to display the plot
    """
    assert dataset.event_time_bins is not None, "event_time_bins not set in dataset"

    bin_counts = np.zeros(dataset.num_bins, dtype=int)
    valid = 0

    for _ in tqdm(range(num_samples), desc="Sampling dataset"):
        idx = np.random.randint(len(dataset))
        item = dataset[idx]
        if item is None:
            continue

        b = item['time_to_event_bin']
        if b is not None:
            bin_counts[b] += 1
            valid += 1

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(dataset.num_bins), bin_counts)
    plt.xlabel("Event-time bin")
    plt.ylabel("Number of samples")
    plt.title(f"Samples per bin (n={valid})")
    plt.grid(axis="y")
    plt.savefig(Path('temp') / 'samples_per_bin.png')


def plot_event_vs_censoring_hist(self,
                                 title='Events vs Censored',
                                 save_path=None,
                                 bins=16,
                                 show=False):
    """
    Plot a histogram of observation times for event vs. censored samples.

    Args:
        save_path (str or Path, optional): Path to save the figure. If None, doesn't save.
        bins (int): Number of histogram bins.
        show (bool): Whether to display the plot.
    """
    observation_times = self.cell_metadata['End Frames'] - self.cell_metadata[
        'Start Frames'] + 1
    death_frames = self.cell_metadata['Death Frames']

    is_event = ~np.isnan(death_frames)
    is_censored = np.isnan(death_frames)

    event_times = observation_times[is_event]
    censor_times = observation_times[is_censored]

    plt.figure(figsize=(10, 6))
    plt.hist(event_times,
             bins=bins,
             alpha=0.7,
             label=f'Deaths {len(event_times)}',
             color='tab:red')
    plt.hist(censor_times,
             bins=bins,
             alpha=0.7,
             label=f'Censored {len(censor_times)}',
             color='tab:blue')
    plt.xlabel('Observation Time (frames)')
    plt.ylabel('Number of Samples')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved event/censoring time histogram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
