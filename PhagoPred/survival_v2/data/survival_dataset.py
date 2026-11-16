from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union
from functools import cached_property

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from .dataset import (CellSample, CellDataset, collate_fn, ArrayType_T,
                      ScalarType_T, PathType_T, ListType_T)

from .graph_synthetic.scenarios import load_scenario
from .graph_synthetic.variances import estimate_unobserved_variance

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
        self._bins = event_time_bins

        super().__init__(*args, **kwargs)

    def _compute_bins(self, num_samples: int = 10000) -> None:
        """Compute event time bins based on quantiles of observed event times in the dataset."""
        times = []
        events = 0
        idxs = self.sample_indices(num_samples)
        for idx in idxs:
            item: SurvivalCellSample = self[idx]

            e = item.event_indicator
            t = item.time_to_event

            events += e
            if e == 1 and t <= self.max_time_to_death:
                times.append(t)

        times = np.array(times)
        bins = np.quantile(times, np.linspace(0, 1,
                                              self.num_bins + 1)).astype(int)
        bins[-1] = self.max_time_to_death + 1
        self.bins = bins

    def _get_equal_bins(self) -> np.ndarray:
        return np.linspace(0, self.max_time_to_death,
                           self.num_bins + 1).astype(int)

    @cached_property
    def bins(self):
        if self._bins is None:
            self._bins = self._get_equal_bins()
        return self._bins

    def get_bin_weights(self, num_samples: int = 1000) -> list:
        """Inverse frequency of bins (for uncensored samles only), for weighting NLL loss."""
        counts = np.zeros(self.num_bins)
        idxs = self.sample_indices(num_samples)
        for idx in idxs:
            # idx = np.random.randint(len(self))
            sample = self[idx]
            if sample.event_indicator == 1:
                counts[sample.time_to_event_bin] += 1
        counts = np.where(counts == 0, 1, counts)
        weights = counts.sum() / (self.num_bins * counts)
        return weights.tolist()

    @property
    def horizon(self):
        return int(self.bins[-1])

    @staticmethod
    def landmarks_rule(start_frame, last_frame, death_frame, min_len, horizon):
        last = last_frame if np.isnan(death_frame) else death_frame
        return np.arange(int(start_frame + min_len), int(last))

    def landmark_labels(self, idx: int,
                        landmarks: np.ndarray) -> dict[str, np.ndarray | None]:
        """Labels at each landmark L:

        * time_to_event: frames from L to death (or to the last observed
          frame if censored)
        * time_to_event_bin: its bin in event_time_bins; times beyond the last
          bin are censored there, with bin = num_bins (None without bins)
        * event_indicator: 1 if death is observed within the bins, else 0
        * binned_pmf: true P(death in each bin | alive at L), (L, num_bins),
          from the stored PMF; None without one (real data) or without bins
        """
        landmarks = np.asarray(landmarks, dtype=int)
        meta = self.cell_metadata
        death = float(meta.death_frames[idx])
        censored = np.isnan(death)
        last_frame = float(meta.end_frames[idx]) if censored else death
        time_to_event = last_frame - landmarks
        event_indicator = np.full(len(landmarks), 0 if censored else 1)

        time_to_event_bin = None
        binned_pmf = None
        bins = self.bins
        if bins is not None:
            within = time_to_event <= bins[-1]
            time_to_event_bin = np.where(
                within,
                np.minimum(np.digitize(time_to_event, bins[1:]),
                           self.num_bins - 1), self.num_bins)
            event_indicator = np.where(within, event_indicator, 0)

            pmf = self._cell_pmf(idx)
            if pmf is not None:
                # cif0[k] = P(death before frame k), so a bin [a, b) relative
                # to L has mass cif0[L + b] - cif0[L + a] (clipped to the end).
                cif0 = np.concatenate([[0.0],
                                       np.cumsum(pmf, dtype=np.float64)])
                end = len(pmf)
                survival = np.clip(1.0 - cif0[landmarks], 0.0, 1.0)
                edges = np.minimum(landmarks[:, None] +
                                   bins[None, :].astype(int),
                                   end)  # (L, num_bins + 1)
                mass = np.diff(cif0[edges], axis=1)  # (L, num_bins)
                binned_pmf = np.divide(mass,
                                       survival[:, None],
                                       out=np.zeros_like(mass),
                                       where=survival[:, None] > 0.0)
                binned_pmf = np.clip(np.round(binned_pmf, decimals=4), 0.0,
                                     1.0)

        return {
            'time_to_event': time_to_event,
            'time_to_event_bin': time_to_event_bin,
            'event_indicator': event_indicator,
            'binned_pmf': binned_pmf,
        }

    def __getitem__(self, idx: int) -> SurvivalCellSample:
        """One sample: cell ``idx`` at a random valid landmark."""
        meta = self.cell_metadata
        features, observed = self.cell_history(idx)
        landmark = int(np.random.choice(self.valid_landmarks(idx)))
        labels = self.landmark_labels(idx, [landmark])
        bin_ = labels['time_to_event_bin']
        binned = labels['binned_pmf']
        start_frame = meta.start_frames[idx]

        return SurvivalCellSample(
            features=features,
            mask=observed,
            length=landmark - start_frame,
            cell_idx=meta.local_cell_idxs[idx],
            hdf5_path=self.hdf5_paths[meta.file_idxs[idx]],
            start_frame=start_frame,
            landmark_frame=landmark,
            last_frame=meta.end_frames[idx],
            time_to_event=labels['time_to_event'][0],
            time_to_event_bin=None if bin_ is None else int(bin_[0]),
            event_indicator=int(labels['event_indicator'][0]),
            pmf=self._landmark_pmf(idx, landmark),
            binned_pmf=None if binned is None else binned[0])

    def _landmark_pmf(self, idx: int, landmark: int) -> np.ndarray | None:
        """True per-frame PMF from ``landmark`` on, given alive at it (sampled
        view only: its length varies with the landmark)."""
        pmf = self._cell_pmf(idx)
        if pmf is None or self.bins is None:
            return None
        survival = float(np.clip(1.0 - pmf[:landmark].sum(), 0.0, 1.0))
        if survival <= 0.0:
            return pmf[landmark:] * 0.0
        return np.clip(pmf[landmark:] / survival, 0.0, 1.0)

    @cached_property
    def unobserved_variance(self):
        v = self._get_unobserved_variances()
        if v is None:
            return None
        else:
            return v.pmf

    @cached_property
    def total_variance(self) -> np.ndarray:
        if any(pmfs is None for pmfs in self._pmfs_cache):
            return None
        pmfs = [
            self.landmark_labels(idx, self.valid_landmarks(idx))['binned_pmf']
            for idx in range(len(self))
        ]
        return np.var(np.concatenate(pmfs, axis=0), axis=0)


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
    assert dataset.bins is not None, "event_time_bins not set in dataset"

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
