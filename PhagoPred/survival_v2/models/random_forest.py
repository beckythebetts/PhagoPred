"""
Random Forest wrapper for survival analysis.
Treats survival prediction as multi-class classification over discrete time bins.
"""

from typing import Dict, List, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .classical_base import ClassicalSurvivalModel


class RandomForestSurvival(ClassicalSurvivalModel):
    """
    Random Forest classifier for discrete-time survival analysis.

    Predicts probability mass function (PMF) over discrete time bins
    using a Random Forest classifier with class probabilities.

    Uses IPCW (Inverse Probability of Censoring Weighting) to handle
    censored observations during training.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        class_weight: Optional[str] = 'balanced',
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        feature_extraction: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize Random Forest survival model.

        Args:
            num_bins: Number of discrete time bins
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features for best split ('sqrt', 'log2', or int)
            bootstrap: Whether to use bootstrap samples
            class_weight: Weight balancing strategy ('balanced' or None)
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            feature_extraction: Temporal feature extraction method
            n_segments: Number of segments for feature extraction
            n_lags: Number of lag features
            window_sizes: Window sizes for rolling features
            feature_names: Names of input features
        """
        super().__init__(
            num_bins=num_bins,
            feature_extraction=feature_extraction,
            n_segments=n_segments,
            n_lags=n_lags,
            window_sizes=window_sizes,
            feature_names=feature_names
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """
        Fit the Random Forest model.

        Uses IPCW weighting for censored observations:
        - Observed events: full weight
        - Censored observations: weighted by censoring probability

        Args:
            X: Tabular features (n_samples, n_features)
            y: Time bin indices (n_samples,)
            events: Event indicators (n_samples,) - 1 for event, 0 for censored

        Returns:
            Training history dictionary
        """
        # Compute sample weights using IPCW
        sample_weights = self._compute_ipcw_weights(y, events)

        # Ensure all classes are present (0 to num_bins-1)
        # If some bins have no samples, we still need to handle them
        classes = np.arange(self.num_bins)

        # For uncensored observations, use their actual time bin
        # For censored observations, we have two strategies:
        # 1. Exclude them (loses information)
        # 2. Use IPCW weighting (our approach)

        # Fit on all data with IPCW weights
        self.model.fit(X, y, sample_weight=sample_weights)

        # Store classes for prediction
        self._classes = classes

        return {
            'n_samples': len(y),
            'n_events': events.sum(),
            'n_censored': len(events) - events.sum(),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _compute_ipcw_weights(self, y: np.ndarray, events: np.ndarray) -> np.ndarray:
        """
        Compute Inverse Probability of Censoring Weights.

        For uncensored observations: weight = 1
        For censored observations: weight based on censoring probability

        A simple approach: use 1 for events, and a reduced weight for censored.
        More sophisticated IPCW would estimate the censoring distribution.
        """
        weights = np.ones(len(y))

        # Simple approach: reduce weight for censored observations
        # This is a simplified IPCW - proper implementation would estimate
        # the censoring survival function using Kaplan-Meier
        censored_mask = events == 0

        # Censored observations get lower weight
        # Weight decreases for earlier censoring (less informative)
        # Normalize time to 0-1 range and use as weight
        if censored_mask.sum() > 0:
            max_time = y.max() if y.max() > 0 else 1
            weights[censored_mask] = (y[censored_mask] + 1) / (max_time + 1)

        return weights

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict PMF over time bins.

        Args:
            X: Tabular features (n_samples, n_features)

        Returns:
            PMF predictions (n_samples, num_bins)
        """
        # Get class probabilities
        proba = self.model.predict_proba(X)

        # Handle case where not all classes were seen during training
        if proba.shape[1] != self.num_bins:
            full_proba = np.zeros((X.shape[0], self.num_bins))
            for i, c in enumerate(self.model.classes_):
                if c < self.num_bins:
                    full_proba[:, c] = proba[:, i]
            # Normalize to sum to 1
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            proba = full_proba / row_sums

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before getting importance")

        feature_names = self.get_feature_names()
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))
