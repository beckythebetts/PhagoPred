"""
Temporal feature extraction for converting sequences to tabular format.
Enables use of traditional ML models (Random Forest, XGBoost, etc.) on sequential data.
"""

import numpy as np
from typing import List, Optional, Tuple, Union


class TemporalFeatureExtractor:
    """
    Extract tabular features from temporal sequences.

    Supports multiple extraction methods that capture different aspects
    of the temporal dynamics in the data.

    Methods:
        - 'mean': Simple mean over time (1 value per feature)
        - 'last': Last valid timestep values (1 value per feature)
        - 'summary': Statistics: mean, std, min, max, first, last, slope (7 per feature)
        - 'segments': Divide into N segments, compute stats per segment (2*N per feature)
        - 'lags': Last N timestep values flattened (N per feature)
        - 'distributed_lags': Sample values at N proportional positions (N per feature)
        - 'rolling': Rolling window stats at different scales (3*windows per feature)
        - 'temporal_full': All methods combined (~36 per feature)
    """

    def __init__(
        self,
        method: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        lag_positions: List[float] = None,  # For distributed_lags: proportions like [0, 0.25, 0.5, 0.75, 1.0]
        window_sizes: List[int] = None,
        feature_names: List[str] = None
    ):
        """
        Initialize the feature extractor.

        Args:
            method: Feature extraction method. One of:
                'mean', 'last', 'summary', 'segments', 'lags', 'distributed_lags', 'rolling', 'temporal_full'
            n_segments: Number of temporal segments for 'segments' method
            n_lags: Number of lag features for 'lags' method
            lag_positions: Proportional positions for 'distributed_lags' (0.0=start, 1.0=end)
                Default: [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] - samples throughout sequence
            window_sizes: Window sizes for 'rolling' method (default: [5, 10, 20])
            feature_names: Names of input features (for generating output feature names)
        """
        valid_methods = ['mean', 'last', 'summary', 'segments', 'lags', 'distributed_lags', 'rolling', 'temporal_full']
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'. Must be one of {valid_methods}")

        self.method = method
        self.n_segments = n_segments
        self.n_lags = n_lags
        self.lag_positions = lag_positions or [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        self.window_sizes = window_sizes or [5, 10, 20]
        self.feature_names = feature_names
        self._num_input_features = None
        self._output_feature_names = None

    def fit(self, X: np.ndarray, lengths: np.ndarray) -> 'TemporalFeatureExtractor':
        """
        Fit the extractor (learns number of features, validates input).

        Args:
            X: Input sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            self
        """
        self._num_input_features = X.shape[-1]
        if self.feature_names is None:
            self.feature_names = [f'feat_{i}' for i in range(self._num_input_features)]

        self._output_feature_names = self._generate_feature_names()
        return self

    def transform(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Transform sequences to tabular features.

        Args:
            X: Input sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Tabular features of shape (batch, num_tabular_features)
        """
        if self._num_input_features is None:
            self.fit(X, lengths)

        if self.method == 'mean':
            return self._extract_mean(X, lengths)
        elif self.method == 'last':
            return self._extract_last(X, lengths)
        elif self.method == 'summary':
            return self._extract_summary(X, lengths)
        elif self.method == 'segments':
            return self._extract_segments(X, lengths)
        elif self.method == 'lags':
            return self._extract_lags(X, lengths)
        elif self.method == 'distributed_lags':
            return self._extract_distributed_lags(X, lengths)
        elif self.method == 'rolling':
            return self._extract_rolling(X, lengths)
        elif self.method == 'temporal_full':
            return self._extract_temporal_full(X, lengths)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, lengths)
        return self.transform(X, lengths)

    def get_feature_names(self) -> List[str]:
        """Return names of all output features."""
        if self._output_feature_names is None:
            if self._num_input_features is None:
                raise ValueError("Must call fit() before get_feature_names()")
            self._output_feature_names = self._generate_feature_names()
        return self._output_feature_names

    def get_num_output_features(self, num_input_features: int) -> int:
        """Calculate number of output features given input feature count."""
        if self.method == 'mean':
            return num_input_features
        elif self.method == 'last':
            return num_input_features
        elif self.method == 'summary':
            return num_input_features * 7  # mean, std, min, max, first, last, slope
        elif self.method == 'segments':
            return num_input_features * self.n_segments * 2  # mean, std per segment
        elif self.method == 'lags':
            return num_input_features * self.n_lags
        elif self.method == 'distributed_lags':
            return num_input_features * len(self.lag_positions)
        elif self.method == 'rolling':
            return num_input_features * len(self.window_sizes) * 3  # mean, std, slope per window
        elif self.method == 'temporal_full':
            # summary (7) + segments (n_segments*2) + rolling (windows*3) + lags (n_lags)
            return num_input_features * (7 + self.n_segments * 2 + len(self.window_sizes) * 3 + self.n_lags)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_feature_names(self) -> List[str]:
        """Generate names for all output features."""
        names = []

        if self.method == 'mean':
            names = [f'{fn}_mean' for fn in self.feature_names]

        elif self.method == 'last':
            names = [f'{fn}_last' for fn in self.feature_names]

        elif self.method == 'summary':
            for fn in self.feature_names:
                names.extend([
                    f'{fn}_mean', f'{fn}_std', f'{fn}_min', f'{fn}_max',
                    f'{fn}_first', f'{fn}_last', f'{fn}_slope'
                ])

        elif self.method == 'segments':
            for fn in self.feature_names:
                for s in range(self.n_segments):
                    names.extend([f'{fn}_seg{s}_mean', f'{fn}_seg{s}_std'])

        elif self.method == 'lags':
            for fn in self.feature_names:
                for lag in range(self.n_lags):
                    names.append(f'{fn}_lag{lag}')

        elif self.method == 'distributed_lags':
            for fn in self.feature_names:
                for pos in self.lag_positions:
                    names.append(f'{fn}_pos{int(pos*100)}pct')

        elif self.method == 'rolling':
            for fn in self.feature_names:
                for ws in self.window_sizes:
                    names.extend([
                        f'{fn}_roll{ws}_mean', f'{fn}_roll{ws}_std', f'{fn}_roll{ws}_slope'
                    ])

        elif self.method == 'temporal_full':
            for fn in self.feature_names:
                # Summary stats
                names.extend([
                    f'{fn}_mean', f'{fn}_std', f'{fn}_min', f'{fn}_max',
                    f'{fn}_first', f'{fn}_last', f'{fn}_slope'
                ])
                # Segment stats
                for s in range(self.n_segments):
                    names.extend([f'{fn}_seg{s}_mean', f'{fn}_seg{s}_std'])
                # Rolling stats
                for ws in self.window_sizes:
                    names.extend([
                        f'{fn}_roll{ws}_mean', f'{fn}_roll{ws}_std', f'{fn}_roll{ws}_slope'
                    ])
                # Lag features
                for lag in range(self.n_lags):
                    names.append(f'{fn}_lag{lag}')

        return names

    def _extract_mean(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract mean over valid timesteps."""
        batch_size, seq_len, num_features = X.shape
        result = np.zeros((batch_size, num_features))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                result[i] = np.nanmean(X[i, :valid_len], axis=0)

        return result

    def _extract_last(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract last valid timestep values."""
        batch_size, seq_len, num_features = X.shape
        result = np.zeros((batch_size, num_features))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                result[i] = X[i, valid_len - 1]

        return result

    def _extract_summary(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract summary statistics: mean, std, min, max, first, last, slope."""
        batch_size, seq_len, num_features = X.shape
        result = np.zeros((batch_size, num_features * 7))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                seq = X[i, :valid_len]  # (valid_len, num_features)

                for f in range(num_features):
                    feat_seq = seq[:, f]
                    valid_mask = ~np.isnan(feat_seq)

                    if valid_mask.sum() > 0:
                        valid_vals = feat_seq[valid_mask]
                        base_idx = f * 7

                        result[i, base_idx] = np.mean(valid_vals)
                        result[i, base_idx + 1] = np.std(valid_vals) if len(valid_vals) > 1 else 0
                        result[i, base_idx + 2] = np.min(valid_vals)
                        result[i, base_idx + 3] = np.max(valid_vals)
                        result[i, base_idx + 4] = valid_vals[0]
                        result[i, base_idx + 5] = valid_vals[-1]

                        # Slope via linear regression
                        if len(valid_vals) > 1:
                            t = np.arange(len(valid_vals))
                            slope = np.polyfit(t, valid_vals, 1)[0]
                            result[i, base_idx + 6] = slope

        return result

    def _extract_segments(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract stats per temporal segment."""
        batch_size, seq_len, num_features = X.shape
        result = np.zeros((batch_size, num_features * self.n_segments * 2))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                seq = X[i, :valid_len]
                segment_size = max(1, valid_len // self.n_segments)

                for f in range(num_features):
                    for s in range(self.n_segments):
                        start = s * segment_size
                        end = min((s + 1) * segment_size, valid_len) if s < self.n_segments - 1 else valid_len

                        if start < end:
                            segment = seq[start:end, f]
                            valid_mask = ~np.isnan(segment)

                            if valid_mask.sum() > 0:
                                valid_vals = segment[valid_mask]
                                base_idx = (f * self.n_segments + s) * 2
                                result[i, base_idx] = np.mean(valid_vals)
                                result[i, base_idx + 1] = np.std(valid_vals) if len(valid_vals) > 1 else 0

        return result

    def _extract_lags(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract last N timestep values as lag features."""
        batch_size, seq_len, num_features = X.shape
        result = np.zeros((batch_size, num_features * self.n_lags))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                for f in range(num_features):
                    for lag in range(self.n_lags):
                        idx = valid_len - 1 - lag
                        if idx >= 0:
                            result[i, f * self.n_lags + lag] = X[i, idx, f]

        return result

    def _extract_distributed_lags(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """
        Extract values at proportionally distributed positions throughout the sequence.

        Unlike 'lags' which takes the last N values, this samples at proportional
        positions (e.g., 0%, 25%, 50%, 75%, 100%) to capture patterns from the
        entire sequence, not just the end.

        Args:
            X: Input sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Features of shape (batch, num_features * len(lag_positions))
        """
        batch_size, seq_len, num_features = X.shape
        n_positions = len(self.lag_positions)
        result = np.zeros((batch_size, num_features * n_positions))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                for f in range(num_features):
                    for p_idx, pos in enumerate(self.lag_positions):
                        # Convert proportion to index
                        # pos=0.0 -> index 0 (start)
                        # pos=1.0 -> index valid_len-1 (end)
                        idx = int(pos * (valid_len - 1)) if valid_len > 1 else 0
                        idx = min(max(idx, 0), valid_len - 1)  # Clamp to valid range

                        value = X[i, idx, f]
                        if not np.isnan(value):
                            result[i, f * n_positions + p_idx] = value

        return result

    def _extract_rolling(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract rolling window statistics at the end of sequence."""
        batch_size, seq_len, num_features = X.shape
        n_stats = 3  # mean, std, slope
        result = np.zeros((batch_size, num_features * len(self.window_sizes) * n_stats))

        for i in range(batch_size):
            valid_len = int(lengths[i])
            if valid_len > 0:
                seq = X[i, :valid_len]

                for f in range(num_features):
                    for w_idx, ws in enumerate(self.window_sizes):
                        start = max(0, valid_len - ws)
                        window = seq[start:valid_len, f]
                        valid_mask = ~np.isnan(window)

                        if valid_mask.sum() > 0:
                            valid_vals = window[valid_mask]
                            base_idx = (f * len(self.window_sizes) + w_idx) * n_stats

                            result[i, base_idx] = np.mean(valid_vals)
                            result[i, base_idx + 1] = np.std(valid_vals) if len(valid_vals) > 1 else 0

                            if len(valid_vals) > 1:
                                t = np.arange(len(valid_vals))
                                slope = np.polyfit(t, valid_vals, 1)[0]
                                result[i, base_idx + 2] = slope

        return result

    def _extract_temporal_full(self, X: np.ndarray, lengths: np.ndarray) -> np.ndarray:
        """Extract all temporal features combined."""
        summary = self._extract_summary(X, lengths)
        segments = self._extract_segments(X, lengths)
        rolling = self._extract_rolling(X, lengths)
        lags = self._extract_lags(X, lengths)

        # Interleave features per input feature for better interpretability
        batch_size = X.shape[0]
        num_features = X.shape[-1]

        n_summary = 7
        n_segments = self.n_segments * 2
        n_rolling = len(self.window_sizes) * 3
        n_lags = self.n_lags
        n_per_feature = n_summary + n_segments + n_rolling + n_lags

        result = np.zeros((batch_size, num_features * n_per_feature))

        for f in range(num_features):
            out_start = f * n_per_feature

            # Summary stats
            result[:, out_start:out_start + n_summary] = summary[:, f * n_summary:(f + 1) * n_summary]
            out_start += n_summary

            # Segment stats
            result[:, out_start:out_start + n_segments] = segments[:, f * n_segments:(f + 1) * n_segments]
            out_start += n_segments

            # Rolling stats
            result[:, out_start:out_start + n_rolling] = rolling[:, f * n_rolling:(f + 1) * n_rolling]
            out_start += n_rolling

            # Lag features
            result[:, out_start:out_start + n_lags] = lags[:, f * n_lags:(f + 1) * n_lags]

        return result
