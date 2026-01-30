"""
Base class for classical (non-deep learning) survival models.
Provides common interface for sklearn-style models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import DataLoader
import torch

from ..features import TemporalFeatureExtractor


class ClassicalSurvivalModel(ABC):
    """
    Base class for classical (sklearn-style) survival models.

    These models work with tabular data, so they use TemporalFeatureExtractor
    to convert sequential input to tabular features.

    Subclasses must implement:
        - _fit_model(X, y, event): Train the underlying model
        - _predict_proba(X): Predict PMF over time bins
    """

    def __init__(
        self,
        num_bins: int,
        feature_extraction: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize the classical survival model.

        Args:
            num_bins: Number of discrete time bins for prediction
            feature_extraction: Method for temporal feature extraction.
                One of: 'mean', 'last', 'summary', 'segments', 'lags', 'rolling', 'temporal_full'
            n_segments: Number of temporal segments for 'segments' method
            n_lags: Number of lag features for 'lags' method
            window_sizes: Window sizes for 'rolling' method
            feature_names: Names of input features
            **kwargs: Additional arguments passed to subclass
        """
        self.num_bins = num_bins
        self.feature_extraction = feature_extraction
        self.n_segments = n_segments
        self.n_lags = n_lags
        self.window_sizes = window_sizes or [5, 10, 20]
        self.input_feature_names = feature_names

        self.feature_extractor = TemporalFeatureExtractor(
            method=feature_extraction,
            n_segments=n_segments,
            n_lags=n_lags,
            window_sizes=self.window_sizes,
            feature_names=feature_names
        )

        self._is_fitted = False
        self.model = None

    def extract_features(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Convert sequence data to tabular features.

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Tabular features of shape (batch, num_tabular_features)
        """
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if isinstance(lengths, torch.Tensor):
            lengths = lengths.cpu().numpy()

        return self.feature_extractor.transform(x, lengths)

    def fit(self, train_loader: DataLoader, device: str = 'cpu') -> Dict:
        """
        Train the model on data from a DataLoader.

        Args:
            train_loader: DataLoader providing training batches
            device: Device (unused for classical models, kept for API compatibility)

        Returns:
            Dictionary with training history/metrics
        """
        # Extract features batch by batch (handles variable sequence lengths)
        # Each batch may have different padded seq_len, so we extract features
        # per batch and concatenate the fixed-size tabular outputs
        all_tabular_features = []
        all_times = []
        all_events = []
        first_batch = True

        for batch in train_loader:
            features = batch['features']
            lengths = batch['length']
            time_bins = batch['time_to_event_bin']
            events = batch['event_indicator']

            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.cpu().numpy()
            if isinstance(time_bins, torch.Tensor):
                time_bins = time_bins.cpu().numpy()
            if isinstance(events, torch.Tensor):
                events = events.cpu().numpy()

            # Fit feature extractor on first batch to initialize
            if first_batch:
                self.feature_extractor.fit(features, lengths)
                first_batch = False

            # Extract tabular features for this batch
            # This produces fixed-size output regardless of seq_len
            batch_tabular = self.feature_extractor.transform(features, lengths)
            all_tabular_features.append(batch_tabular)
            all_times.append(time_bins)
            all_events.append(events)

        # Concatenate all tabular features (now same dimensions)
        X = np.concatenate(all_tabular_features, axis=0)
        y = np.concatenate(all_times, axis=0)
        events = np.concatenate(all_events, axis=0)

        # Handle any NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit the underlying model
        history = self._fit_model(X, y, events)
        self._is_fitted = True

        return history

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """
        Fit the underlying sklearn model.

        Args:
            X: Tabular features of shape (n_samples, n_features)
            y: Time bin indices of shape (n_samples,)
            events: Event indicators of shape (n_samples,) - 1 for event, 0 for censored

        Returns:
            Dictionary with training history/metrics
        """
        pass

    @abstractmethod
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability mass function over time bins.

        Args:
            X: Tabular features of shape (n_samples, n_features)

        Returns:
            PMF predictions of shape (n_samples, num_bins)
        """
        pass

    def predict_pmf(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict PMF from sequence input.

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            PMF predictions of shape (batch, num_bins)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X = self.extract_features(x, lengths)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._predict_proba(X)

    def predict_cif(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict cumulative incidence function from sequence input.

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            CIF predictions of shape (batch, num_bins)
        """
        pmf = self.predict_pmf(x, lengths)
        return np.cumsum(pmf, axis=-1)

    def predict_survival(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict survival function S(t) = 1 - CIF(t).

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Survival function predictions of shape (batch, num_bins)
        """
        return 1.0 - self.predict_cif(x, lengths)

    def predict_median_time(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict median survival time (bin where CIF crosses 0.5).

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Median time bin predictions of shape (batch,)
        """
        cif = self.predict_cif(x, lengths)
        # Find first bin where CIF >= 0.5
        median_bins = np.argmax(cif >= 0.5, axis=-1).astype(float)
        # Handle cases where CIF never reaches 0.5
        never_reaches = np.all(cif < 0.5, axis=-1)
        median_bins[never_reaches] = self.num_bins - 1
        return median_bins

    def predict_expected_time(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict expected time (weighted average of bins by PMF).

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)

        Returns:
            Expected time predictions of shape (batch,)
        """
        pmf = self.predict_pmf(x, lengths)
        bins = np.arange(self.num_bins)
        return np.sum(pmf * bins, axis=-1)

    def get_feature_names(self) -> List[str]:
        """Get names of extracted tabular features."""
        return self.feature_extractor.get_feature_names()

    def save(self, path: str) -> None:
        """Save the model to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ClassicalSurvivalModel':
        """Load a model from disk."""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
        lengths: Union[np.ndarray, torch.Tensor],
        return_attention: bool = False,
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, None]]:
        """
        Make model callable like PyTorch models.

        Args:
            x: Sequences of shape (batch, seq_len, num_features)
            lengths: Valid lengths for each sequence (batch,)
            return_attention: If True, return (pmf, None) - attention not supported
            mask: Unused, kept for API compatibility

        Returns:
            PMF predictions, or (PMF, None) if return_attention=True
        """
        pmf = self.predict_pmf(x, lengths)
        if return_attention:
            return pmf, None
        return pmf
