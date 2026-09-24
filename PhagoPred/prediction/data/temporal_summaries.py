from __future__ import annotations
from dataclasses import dataclass, fields
from functools import partial
from typing import Union

import numpy as np
from tqdm import tqdm

from PhagoPred.utils.logger import get_logger
from .binary_dataset import BinaryCellDataset, BinaryCellBatch
from .survival_dataset import SurvivalCellBatch
from .dataset import CellDataset

log = get_logger()


@dataclass
class BinaryTemporalSummaryDataset(BinaryCellBatch):
    """Binary Cell Sample class with added field for tmeporal summary."""
    temporal_summary_features: np.ndarray


@dataclass
class SurvivalTemopralSummaryDataset(SurvivalCellBatch):
    """Suvival Cell Sample class with added field for tmeporal summary."""
    temporal_summary_features: np.ndarray


class TemporalSummary:
    """Summary statistics for temporal data.
    Using sliding windows of varying scales.
    """

    def __init__(
        self,
        window_sizes: list,
        window_overlap: float = 0.5,
    ):
        self.window_sizes = window_sizes
        self.window_overlap = window_overlap

        self.base_feature_names = None
        self.feature_names = None

        self.summary_stats = {
            'mean': partial(np.nanmean,
                            axis=1),  #lamda functions cannot be pickled
            'std': partial(np.nanstd, axis=1),
            'slope': self._slope,
            'dom_freq': self._dom_freq,
        }

        self.full_seq_len = None

    def convert_ds(
        self,
        dataset: CellDataset,
        num_landmarks_per_sample: int = 2
    ) -> Union[
            BinaryCellBatch,
            SurvivalCellBatch,
    ]:
        """
        Convert dataset to Batch of all cells with tmeporal summary statistics calucated over differnet sized sliding windows
        """

        log.info(f'Converting dataset {dataset} to temporal summary')
        self.base_feature_names = dataset.feature_names
        self.full_seq_len = dataset.full_len_frames

        progress_bar = tqdm(total=num_landmarks_per_sample * len(dataset),
                            desc="Generating temporal summaries")

        if isinstance(dataset, BinaryCellDataset):
            BatchClass = BinaryTemporalSummaryDataset
        else:
            BatchClass = SurvivalTemopralSummaryDataset

        batch_lists = {f.name: [] for f in fields(BatchClass)}

        for _ in range(num_landmarks_per_sample):
            for idx, _ in enumerate(dataset):
                cell_sample = dataset[idx]
                cell_features = cell_sample.features
                cell_temporal_summary_features = self._apply_sliding_windows(
                    cell_features)

                for field in fields(cell_sample):
                    batch_lists[field.name].append(
                        getattr(cell_sample, field.name))
                batch_lists['temporal_summary_features'].append(
                    cell_temporal_summary_features)

                progress_bar.update(1)
        batch_dict = {}
        for key, value in batch_lists.items():
            if key in ['pmf', 'binned_pmf']:
                batch_dict[key] = value
            else:
                batch_dict[key] = np.array(value)
        batch = BatchClass(**batch_dict)
        log.info('Temporal dataset summary complete')
        return batch

    def get_feature_names(self) -> list[str]:
        """Get the feature names of the dataset."""
        if self.feature_names is None:
            self.feature_names = self._generate_feature_names()
        return self.feature_names

    def _generate_feature_names(self) -> list[str]:
        names = []
        for window_size in self.window_sizes:
            stride = int(window_size * self.window_overlap)
            num_windows = (self.full_seq_len - window_size) // stride + 1
            names.extend(f'missingness_ws{window_size}_wi{i}'
                         for i in range(num_windows))
            for summary_stat in self.summary_stats.keys():
                for window_idx in range(num_windows):
                    for feat in self.base_feature_names:
                        names.append(
                            f'{summary_stat}_{feat}_ws{window_size}_widx{window_idx}'
                        )
        return names

    def _apply_sliding_windows(self,
                               sample: np.ndarray,
                               allow_nans: bool = False) -> np.ndarray:
        """Vectorized sliding window feature extraction.
        Args:
            sample: shape (full_seq_len, num_features)
        Returns:
            all_features: shape (num_extracted_features,)
        """
        all_features = []
        for window_size in self.window_sizes:
            stride = int(window_size * self.window_overlap)
            num_windows = (self.full_seq_len - window_size) // stride + 1

            # Create all windows at once: shape (num_windows, window_size, num_features)
            window_starts = np.arange(num_windows) * stride
            window_idxs = window_starts[:, None] + np.arange(
                window_size)  # (num_windows, window_size)
            windows = sample[
                window_idxs]  # (num_windows, window_size, features)
            # Missingness: fraction of NaN per window (across all features)
            missingness = np.mean(np.isnan(windows),
                                  axis=(1, 2))  # (num_windows,)
            all_features += (list(missingness))

            # Compute summary stats
            for stat_func in self.summary_stats.values():
                stats = stat_func(
                    windows)  # Each is (num_windows, num_features)
                flattened_stats = stats.flatten(
                )  # (num_windows * num_features,)
                if not allow_nans:
                    flattened_stats = np.nan_to_num(flattened_stats, nan=0.0)
                all_features += list(flattened_stats)

        return np.array(all_features)

    def _slope(self, windows: np.ndarray) -> np.ndarray:
        """Vectorized slope calculation for all windows and features.

        Args:
            windows: shape (num_windows, window_size, num_features)
        Returns:
            slopes: shape (num_windows, num_features)
        """
        _, window_size, _ = windows.shape

        t = np.arange(window_size, dtype=np.float64)
        valid = ~np.isnan(windows)  # (num_windows, window_size, num_features)

        n_valid = valid.sum(axis=1)  # (num_windows, num_features)

        windows_filled = np.where(valid, windows, 0.0)
        t_broadcast = t[None, :, None]  # (1, window_size, 1)
        valid_t = np.where(valid, t_broadcast, 0.0)

        sum_t = valid_t.sum(axis=1)  # (num_windows, num_features)
        sum_x = windows_filled.sum(axis=1)  # (num_windows, num_features)
        sum_tx = (valid_t * windows_filled).sum(
            axis=1)  # (num_windows, num_features)
        sum_t2 = (valid_t**2).sum(axis=1)  # (num_windows, num_features)

        numerator = n_valid * sum_tx - sum_t * sum_x
        denominator = n_valid * sum_t2 - sum_t**2

        with np.errstate(divide='ignore', invalid='ignore'):
            slopes = numerator / denominator

        slopes = np.where(n_valid < 2, 0.0, slopes)
        slopes = np.where(n_valid == 0, np.nan, slopes)

        return slopes

    def _missingness(self, x: np.ndarray) -> float:
        return np.mean(np.isnan(x))

    def _dom_freq(self, windows: np.ndarray) -> np.ndarray:
        """Vectorized dominant frequency calculation.
        
        Args:
            windows: shape (num_windows, window_size, num_features)
        Returns:
            dom_freqs: shape (num_windows, num_features)
        """
        _, window_size, _ = windows.shape

        # Mean-center and replace NaN with 0 (neutral after centering)
        means = np.nanmean(windows, axis=1, keepdims=True)
        centered = np.nan_to_num(windows - means, nan=0.0)

        # Vectorized FFT along window dimension
        fft_values = np.fft.rfft(centered, axis=1)
        power_spectrum = np.abs(fft_values)**2

        freqs = np.fft.rfftfreq(window_size)

        dom_idx = np.argmax(power_spectrum[:, 1:, :], axis=1) + 1
        dom_freqs = freqs[dom_idx]

        valid_count = np.sum(~np.isnan(windows), axis=1)
        dom_freqs = np.where(valid_count < 2, 0.0, dom_freqs)
        dom_freqs = np.where(valid_count == 0, np.nan, dom_freqs)

        return dom_freqs
