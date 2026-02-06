import numpy as np
from tqdm import tqdm
from functools import partial


class TemporalSummary:
    """Summary statistics for temporal data.
    Using sliding windows of varying scales.
    """
    def __init__(
        self, 
        window_sizes: list = [25, 100, 500, 1000],
        window_overlap: float = 0.5,
    ):
        self.window_sizes = window_sizes
        self.window_overlap = window_overlap
        
        self.base_feature_names = None
        self.feature_names = None
        
        self.summary_stats = {
            # 'mean': lambda x: np.nanmean(x, axis=1),
            # 'std': lambda x: np.nanstd(x, axis=1),
            'mean': partial(np.nanmean, axis=1), #lamda functions cannot be pickled
            'std': partial(np.nanstd, axis=1),
            'slope': self._slope,
            'dom_freq': self._dom_freq,
        }
        
        self.full_seq_len = None
        
    def convert_ds(self, ds: 'CellDataset', num_landmarks_per_sample: int = 2) -> np.ndarray:
        """

        """
        self.base_feature_names = ds.features
        self.full_seq_len = ds.fixed_len

        features = []
        raw_features = []
        time_to_event = []
        event_indicator = []
        time_to_event_bin = []
        lengths = []

        progress_bar = tqdm(total=num_landmarks_per_sample * len(ds), desc="Generating temporal summaries")
        for _ in range(num_landmarks_per_sample):
            for idx in range(len(ds)):
                sample = ds[idx]
                sample_features = sample['features']
                sample_temporal_summary = self._apply_sliding_windows(sample_features)

                features.append(sample_temporal_summary)
                raw_features.append(sample_features)
                time_to_event.append(sample['time_to_event'])
                event_indicator.append(sample['event_indicator'])
                time_to_event_bin.append(sample['time_to_event_bin'])
                lengths.append(sample['length'])
                progress_bar.update(1)

        progress_bar.close()

        return {
            'features': np.array(features),
            'raw_features': np.array(raw_features),
            'time_to_event': np.array(time_to_event),
            'event_indicator': np.array(event_indicator),
            'time_to_event_bin': np.array(time_to_event_bin),
            'lengths': np.array(lengths),
        }
                         
    def get_feature_names(self) -> list[str]:
        if self.feature_names is None:
            self.feature_names = self._generate_feature_names()
        return self.feature_names
    
    def _generate_feature_names(self) -> list[str]:
        names = []
        for window_size in self.window_sizes:
            stride = int(window_size * self.window_overlap)
            num_windows = (self.full_seq_len - window_size) // stride + 1
            names.append([f'missingneess_ws{window_size}_wi{i}'] for i in range(num_windows))
            for summary_stat in self.summary_stats.keys():
                for window_idx in range(num_windows):
                    for feat in self.base_feature_names:
                        names.append(f'{summary_stat}_{feat}_ws{window_size}_widx{window_idx}')
        return names
    
    def _apply_sliding_windows(self, sample: np.ndarray) -> np.ndarray:
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
            window_idxs = window_starts[:, None] + np.arange(window_size) # (num_windows, window_size)
            windows = sample[window_idxs]  # (num_windows, window_size, features)
            # Missingness: fraction of NaN per window (across all features)
            missingness = np.mean(np.isnan(windows), axis=(1, 2))  # (num_windows,)
            all_features += (list(missingness))
            
            # Compute summary stats
            for stat_func in self.summary_stats.values():
                stats = stat_func(windows)  # Each is (num_windows, num_features)
                flattened_stats = stats.flatten()  # (num_windows * num_features,)
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
        sum_tx = (valid_t * windows_filled).sum(axis=1)  # (num_windows, num_features)
        sum_t2 = (valid_t ** 2).sum(axis=1)  # (num_windows, num_features)

        numerator = n_valid * sum_tx - sum_t * sum_x
        denominator = n_valid * sum_t2 - sum_t ** 2

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
        num_windows, window_size, num_features = windows.shape
        
        # Mean-center and replace NaN with 0 (neutral after centering)
        means = np.nanmean(windows, axis=1, keepdims=True)
        centered = np.nan_to_num(windows - means, nan=0.0)
        
        # Vectorized FFT along window dimension
        fft_values = np.fft.rfft(centered, axis=1)
        power_spectrum = np.abs(fft_values) ** 2

        freqs = np.fft.rfftfreq(window_size)

        dom_idx = np.argmax(power_spectrum[:, 1:, :], axis=1) + 1
        dom_freqs = freqs[dom_idx]

        valid_count = np.sum(~np.isnan(windows), axis=1)
        dom_freqs = np.where(valid_count < 2, 0.0, dom_freqs)
        dom_freqs = np.where(valid_count == 0, np.nan, dom_freqs)
        
        return dom_freqs

if __name__ == "__main__":
    # Test
    from PhagoPred.survival_v2.data.dataset import CellDataset

    ds = CellDataset(
       hdf5_paths=['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_val.h5'],
       fixed_len=1000,
       features = ['0', '1', '2', '3'],
       normalise=False,
    )

    temporal_summary_extractor = TemporalSummary()

    summarised_data = temporal_summary_extractor.convert_ds(ds, num_landmarks_per_sample=2)
    feature_names = temporal_summary_extractor.get_feature_names()

    print("Extracted features shape:", summarised_data['features'].shape)
    # print("Feature names:", feature_names)
        