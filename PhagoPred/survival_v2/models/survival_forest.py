"""
Scikit-survival model wrappers for proper survival analysis.
These models are designed specifically for survival data and handle
censoring natively rather than using IPCW approximations.
"""

from typing import Dict, List, Optional
import numpy as np
from sksurv.ensemble import RandomSurvivalForest

from .classical_base import ClassicalSurvivalModel


class RandomSurvivalForestModel(ClassicalSurvivalModel):
    """
    Random Survival Forest from scikit-survival.

    A proper survival model that natively handles censored observations
    and predicts survival functions rather than class probabilities.

    The survival function is converted to PMF for compatibility with
    the rest of the survival_v2 framework.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize Random Survival Forest model.

        Args:
            num_bins: Number of discrete time bins
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf node
            max_features: Number of features for best split
            n_jobs: Number of parallel jobs
            random_state: Random seed
            feature_extraction: Temporal feature extraction method
            n_segments: Number of segments for feature extraction
            n_lags: Number of lag features
            window_sizes: Window sizes for rolling features
            feature_names: Names of input features
        """
        super().__init__(
            num_bins=num_bins,
            window_sizes=window_sizes,
            feature_names=feature_names
        )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._time_bins = None
        self._init_model()

    def _init_model(self):
        """Initialize the Random Survival Forest model."""

        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )

    def _fit_model(self, train_data: Dict) -> Dict:
        """
        Fit the Random Survival Forest.

        Args:
            X: Tabular features (n_samples, n_features)
            y: Time bin indices (n_samples,)
            events: Event indicators (n_samples,)

        Returns:
            Training history dictionary
        """
        print('Fitting model...')
        features = train_data['features']
        times = train_data['time_to_event_bin']
        events = train_data['event_indicator']
        
        y_structured = np.array(
            [(bool(e), t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)]
        )

        # Fit model
        self.model.fit(features, y_structured)

        # Store the unique event times for prediction
        self._time_bins = np.arange(self.num_bins)

        return {
            'n_samples': len(times),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum())
        }

    # def _predict_proba(self, X: np.ndarray) -> np.ndarray:
    #     """
    #     Predict PMF over time bins.

    #     Converts survival function to PMF:
    #     PMF(t) = S(t-1) - S(t)

    #     Args:
    #         X: Tabular features (n_samples, n_features)

    #     Returns:
    #         PMF predictions (n_samples, num_bins)
    #     """
    #     # Get survival functions
    #     surv_funcs = self.model.predict_survival_function(X)

    #     # Convert to PMF
    #     pmf = np.zeros((X.shape[0], self.num_bins))

    #     for i, sf in enumerate(surv_funcs):
    #         # Evaluate survival function at each bin
    #         surv_at_bins = np.array([sf(t) for t in self._time_bins])

    #         # PMF is the negative derivative of survival function
    #         # PMF(t) = S(t-1) - S(t)
    #         pmf[i, 0] = 1.0 - surv_at_bins[0]
    #         pmf[i, 1:] = surv_at_bins[:-1] - surv_at_bins[1:]

    #         # Ensure non-negative and normalize
    #         pmf[i] = np.maximum(pmf[i], 0)
    #         if pmf[i].sum() > 0:
    #             pmf[i] /= pmf[i].sum()
    #         else:
    #             # Fallback: uniform distribution
    #             pmf[i] = 1.0 / self.num_bins

    #     return pmf

    def _predict_pmfs(self, input_features: np.ndarray):
        """PRedict PMFS from model
        Args:
            input_features: Tabular features (n_samples, n_features)
        Returns:
            PMF predictions (n_samples, num_bins)
        """
        survival_funcs = self.model.predict_survival_function(input_features, return_array=True)
        print(input_features.shape, survival_funcs.shape)
        # P(T=t) = S(t-1) - S(t)
        pmf = survival_funcs[:, :-1] - survival_funcs[:, 1:]
        pmf = np.concatenate((1-survival_funcs[:, 0][:, np.newaxis], pmf), axis=1)
        return pmf #(n_smaples, num_bins)
        
        
    # def predict_survival_function(self, x, lengths) -> List:
    #     """
    #     Get the full survival function predictions.

    #     Args:
    #         x: Sequences of shape (batch, seq_len, num_features)
    #         lengths: Valid lengths for each sequence (batch,)

    #     Returns:
    #         List of survival function objects from scikit-survival
    #     """
    #     if not self._is_fitted:
    #         raise RuntimeError("Model must be fitted first")

    #     X = self.extract_features(x, lengths)
    #     X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    #     return self.model.predict_survival_function(X)


class GradientBoostingSurvivalModel(ClassicalSurvivalModel):
    """
    Gradient Boosting Survival Analysis from scikit-survival.

    Uses a gradient boosting approach with the Cox partial likelihood
    or other survival-specific loss functions.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        dropout_rate: float = 0.0,
        random_state: Optional[int] = 42,
        feature_extraction: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize Gradient Boosting Survival model.

        Args:
            num_bins: Number of discrete time bins
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of individual trees
            learning_rate: Learning rate shrinks contribution of each tree
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            subsample: Fraction of samples for fitting each tree
            dropout_rate: Dropout rate for regularization
            random_state: Random seed
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
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.dropout_rate = dropout_rate
        self.random_state = random_state

        self._time_bins = None
        self._init_model()

    def _init_model(self):
        """Initialize the Gradient Boosting Survival model."""
        try:
            from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        except ImportError:
            raise ImportError(
                "scikit-survival is required for GradientBoostingSurvivalModel. "
                "Install it with: pip install scikit-survival"
            )

        self.model = GradientBoostingSurvivalAnalysis(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            dropout_rate=self.dropout_rate,
            random_state=self.random_state
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """
        Fit the Gradient Boosting Survival model.

        Args:
            X: Tabular features (n_samples, n_features)
            y: Time bin indices (n_samples,)
            events: Event indicators (n_samples,)

        Returns:
            Training history dictionary
        """
        # Create structured array for scikit-survival
        y_structured = np.array(
            [(bool(e), t) for e, t in zip(events, y)],
            dtype=[('event', bool), ('time', float)]
        )

        # Fit model
        self.model.fit(X, y_structured)

        # Store time bins
        self._time_bins = np.arange(self.num_bins)

        return {
            'n_samples': len(y),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict PMF over time bins.

        Args:
            X: Tabular features (n_samples, n_features)

        Returns:
            PMF predictions (n_samples, num_bins)
        """
        # Get survival functions
        surv_funcs = self.model.predict_survival_function(X)

        # Convert to PMF
        pmf = np.zeros((X.shape[0], self.num_bins))

        for i, sf in enumerate(surv_funcs):
            # Evaluate survival function at each bin
            surv_at_bins = np.array([sf(t) for t in self._time_bins])

            # PMF is the negative derivative of survival function
            pmf[i, 0] = 1.0 - surv_at_bins[0]
            pmf[i, 1:] = surv_at_bins[:-1] - surv_at_bins[1:]

            # Ensure non-negative and normalize
            pmf[i] = np.maximum(pmf[i], 0)
            if pmf[i].sum() > 0:
                pmf[i] /= pmf[i].sum()
            else:
                pmf[i] = 1.0 / self.num_bins

        return pmf

    def predict_survival_function(self, x, lengths) -> List:
        """Get full survival function predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        X = self.extract_features(x, lengths)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_survival_function(X)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = self.get_feature_names()
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))
