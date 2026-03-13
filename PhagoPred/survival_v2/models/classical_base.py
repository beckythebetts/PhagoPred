"""
Base class for classical (non-deep learning) survival models.
Provides common interface for sklearn-style models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np
import torch
import pickle

from PhagoPred.survival_v2.data import TemporalSummary, CellDataset, BinaryTemporalSummaryDataset, SurvivalTemopralSummaryDataset


class ClassicalSurvivalModel(ABC):
    """
    Base class for classical (sklearn-style) survival models.

    These models work with tabular data, so they use TemporalFeatureExtractor
    to convert sequential input to tabular features.

    Subclasses must implement:
        - _fit_model(X, y, event): Train the underlying model
        - _predict_proba(X): Predict PMF over time bins
    """

    def __init__(self,
                 num_bins: int,
                 window_sizes: List[int] = None,
                 feature_names: List[str] = None,
                 **kwargs):
        """
        Initialize the classical survival model.

        Args:
            num_bins: Number of discrete time bins for prediction
            feature_extraction: Method for temporal feature extraction.
                One of: 'mean', 'last', 'summary', 'segments', 'lags', 'rolling', 'temporal_full'
            window_sizes: Window sizes for 'rolling' method
            feature_names: Names of input features
            **kwargs: Additional arguments passed to subclass
        """
        self.num_bins = num_bins

        self.binary = True if num_bins == 1 else False
        self.window_sizes = window_sizes
        self.input_feature_names = feature_names

        self.temporal_summariser = TemporalSummary(
            window_sizes=self.window_sizes)

        self._is_fitted = False
        self.model = None

        self.test_data = None

    def get_temporal_summary(
        self, dataset: CellDataset
    ) -> Union[BinaryTemporalSummaryDataset, SurvivalTemopralSummaryDataset]:
        """Get tempororal summary veriosn of dataset."""
        return self.temporal_summariser.convert_ds(dataset)

    def fit(self,
            dataset: Union[CellDataset, torch.utils.data.DataLoader],
            device: str = 'cpu') -> Dict:
        """Fit model from CellDataset"""
        if isinstance(dataset, torch.utils.data.DataLoader):
            dataset = dataset.dataset
        temporal_summary_dataset = self.get_temporal_summary(dataset)
        return self._fit_model(temporal_summary_dataset)

    @abstractmethod
    def _fit_model(
        self, dataset: Union[BinaryTemporalSummaryDataset,
                             SurvivalTemopralSummaryDataset]
    ) -> Dict:
        pass

    def predict(
        self, dataset: Union[BinaryTemporalSummaryDataset,
                             SurvivalTemopralSummaryDataset]
    ) -> np.ndarray:
        """Predict ouput pmf/event probabilities from model
        
        Return:
            pmfs [samples, output_bins] (if survival model)
            probabilities [samples] (if binary model)
        """
        predictions = self._predict(dataset.temporal_summary_features)
        if self.binary:
            predictions = predictions[:, 0]
        return predictions

    @abstractmethod
    def _predict(self, input_features: np.ndarray) -> np.ndarray:
        pass

    # @abstractmethod
    # def _predict_binary(self, input_features: np.ndarray) -> np.ndarray:
    #     pass

    # def predict_cif(self, dataset: Union[CellDataset, dict, np.ndarray]) -> np.ndarray:
    #     return np.cumsum(self.predict_pmfs(dataset), axis=-1)

    # def predict_survival(self, dataset: Union[CellDataset, dict, np.ndarray]) -> np.ndarray:
    #     return 1.0 - self.predict_cif(dataset)

    # def predict_expected_survival_time(self, dataset: Union[CellDataset, dict, np.ndarray]) -> np.ndarray:
    #     pmf = self.predict_pmfs(dataset)
    #     bins = np.arange(self.num_bins)
    #     return np.sum(pmf * bins, axis=-1)

    def get_feature_names(self) -> List[str]:
        """Get names of extracted tabular features."""
        return self.temporal_summariser.get_feature_names()

    def save(self, path: str) -> None:
        """Save the model to disk."""

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'ClassicalSurvivalModel':
        """Load a model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
