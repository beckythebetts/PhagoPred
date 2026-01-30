"""
Gradient Boosting wrappers for survival analysis.
Includes XGBoost and optional LightGBM implementations.
"""

from typing import Dict, List, Optional
import numpy as np
import warnings

from .classical_base import ClassicalSurvivalModel


class XGBoostSurvival(ClassicalSurvivalModel):
    """
    XGBoost classifier for discrete-time survival analysis.

    Predicts probability mass function (PMF) over discrete time bins
    using XGBoost with multi-class classification.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        n_jobs: int = -1,
        random_state: Optional[int] = 42,
        feature_extraction: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize XGBoost survival model.

        Args:
            num_bins: Number of discrete time bins
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Boosting learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns per tree
            reg_alpha: L1 regularization on weights
            reg_lambda: L2 regularization on weights
            min_child_weight: Minimum sum of instance weight in a child
            gamma: Minimum loss reduction for further partition
            n_jobs: Number of parallel threads
            random_state: Random seed
            feature_extraction: Temporal feature extraction method
            n_segments: Number of segments for feature extraction
            n_lags: Number of lag features
            window_sizes: Window sizes for rolling features
            feature_names: Names of input features
            use_gpu: Whether to use GPU acceleration
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
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_gpu = use_gpu

        self._init_model()

    def _init_model(self):
        """Initialize the XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost is required for XGBoostSurvival. "
                "Install it with: pip install xgboost"
            )

        tree_method = 'gpu_hist' if self.use_gpu else 'hist'

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            tree_method=tree_method,
            objective='multi:softprob',
            num_class=self.num_bins,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """
        Fit the XGBoost model.

        Args:
            X: Tabular features (n_samples, n_features)
            y: Time bin indices (n_samples,)
            events: Event indicators (n_samples,)

        Returns:
            Training history dictionary
        """
        # Compute sample weights using IPCW
        sample_weights = self._compute_ipcw_weights(y, events)

        # Fit model
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            verbose=False
        )

        return {
            'n_samples': len(y),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _compute_ipcw_weights(self, y: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Compute IPCW weights (same as RandomForest)."""
        weights = np.ones(len(y))
        censored_mask = events == 0

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
        proba = self.model.predict_proba(X)

        # Ensure output has correct shape
        if proba.shape[1] != self.num_bins:
            full_proba = np.zeros((X.shape[0], self.num_bins))
            for i, c in enumerate(self.model.classes_):
                if c < self.num_bins:
                    full_proba[:, c] = proba[:, i]
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            proba = full_proba / row_sums

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = self.get_feature_names()
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))


class LightGBMSurvival(ClassicalSurvivalModel):
    """
    LightGBM classifier for discrete-time survival analysis.

    Faster alternative to XGBoost, particularly for large datasets.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: int = -1,
        num_leaves: int = 31,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_child_samples: int = 20,
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
        Initialize LightGBM survival model.

        Args:
            num_bins: Number of discrete time bins
            n_estimators: Number of boosting iterations
            max_depth: Maximum depth (-1 for no limit)
            num_leaves: Maximum number of leaves per tree
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training data
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            min_child_samples: Minimum samples in a leaf
            n_jobs: Number of parallel threads
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
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_samples = min_child_samples
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._init_model()

    def _init_model(self):
        """Initialize the LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is required for LightGBMSurvival. "
                "Install it with: pip install lightgbm"
            )

        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_samples=self.min_child_samples,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            objective='multiclass',
            num_class=self.num_bins,
            verbose=-1
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """Fit the LightGBM model."""
        sample_weights = self._compute_ipcw_weights(y, events)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X, y, sample_weight=sample_weights)

        return {
            'n_samples': len(y),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _compute_ipcw_weights(self, y: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Compute IPCW weights."""
        weights = np.ones(len(y))
        censored_mask = events == 0

        if censored_mask.sum() > 0:
            max_time = y.max() if y.max() > 0 else 1
            weights[censored_mask] = (y[censored_mask] + 1) / (max_time + 1)

        return weights

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict PMF over time bins."""
        proba = self.model.predict_proba(X)

        if proba.shape[1] != self.num_bins:
            full_proba = np.zeros((X.shape[0], self.num_bins))
            for i, c in enumerate(self.model.classes_):
                if c < self.num_bins:
                    full_proba[:, c] = proba[:, i]
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            proba = full_proba / row_sums

        return proba

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = self.get_feature_names()
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))


class GradientBoostingSurvival(ClassicalSurvivalModel):
    """
    Scikit-learn Gradient Boosting classifier for survival analysis.

    Fallback option when XGBoost/LightGBM are not available.
    """

    def __init__(
        self,
        num_bins: int,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        feature_extraction: str = 'temporal_full',
        n_segments: int = 5,
        n_lags: int = 10,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """Initialize sklearn GradientBoosting model."""
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
        self.subsample = subsample
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def _fit_model(self, X: np.ndarray, y: np.ndarray, events: np.ndarray) -> Dict:
        """Fit the GradientBoosting model."""
        sample_weights = self._compute_ipcw_weights(y, events)
        self.model.fit(X, y, sample_weight=sample_weights)

        return {
            'n_samples': len(y),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _compute_ipcw_weights(self, y: np.ndarray, events: np.ndarray) -> np.ndarray:
        """Compute IPCW weights."""
        weights = np.ones(len(y))
        censored_mask = events == 0

        if censored_mask.sum() > 0:
            max_time = y.max() if y.max() > 0 else 1
            weights[censored_mask] = (y[censored_mask] + 1) / (max_time + 1)

        return weights

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict PMF over time bins."""
        proba = self.model.predict_proba(X)

        if proba.shape[1] != self.num_bins:
            full_proba = np.zeros((X.shape[0], self.num_bins))
            for i, c in enumerate(self.model.classes_):
                if c < self.num_bins:
                    full_proba[:, c] = proba[:, i]
            row_sums = full_proba.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            proba = full_proba / row_sums

        return proba
