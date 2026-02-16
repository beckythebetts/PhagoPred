"""
Scikit-survival model wrappers for proper survival analysis.
These models are designed specifically for survival data and handle
censoring natively rather than using IPCW approximations.
"""

from typing import Dict, List, Optional
import warnings
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

    A proper survival model that natively handles censored observations
    using gradient boosting with the Cox partial likelihood.

    The survival function is converted to PMF for compatibility with
    the rest of the survival_v2 framework.
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
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

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

    def _fit_model(self, train_data: Dict) -> Dict:
        """
        Fit the Gradient Boosting Survival model.

        Args:
            train_data: Dictionary with keys 'features', 'time_to_event_bin', 'event_indicator'

        Returns:
            Training history dictionary
        """
        print('Fitting model...')
        features = train_data['features']
        times = train_data['time_to_event_bin']
        events = train_data['event_indicator']

        # Create structured array for scikit-survival
        y_structured = np.array(
            [(bool(e), t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)]
        )

        # Fit model
        self.model.fit(features, y_structured)

        # Store time bins
        self._time_bins = np.arange(self.num_bins)

        return {
            'n_samples': len(times),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'feature_importances': self.model.feature_importances_.tolist()
        }

    def _predict_pmfs(self, input_features: np.ndarray) -> np.ndarray:
        """
        Predict PMFs from model.

        Args:
            input_features: Tabular features (n_samples, n_features)

        Returns:
            PMF predictions (n_samples, num_bins)
        """
        survival_funcs = self.model.predict_survival_function(input_features, return_array=True)
        print(input_features.shape, survival_funcs.shape)
        # P(T=t) = S(t-1) - S(t)
        pmf = survival_funcs[:, :-1] - survival_funcs[:, 1:]
        pmf = np.concatenate((1 - survival_funcs[:, 0][:, np.newaxis], pmf), axis=1)
        return pmf  # (n_samples, num_bins)

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = self.get_feature_names()
        importances = self.model.feature_importances_

        return dict(zip(feature_names, importances))


class CoxPHModel(ClassicalSurvivalModel):
    """
    Cox Proportional Hazards model from scikit-survival.

    The classic semi-parametric survival model that models the hazard as:
        h(t|X) = h0(t) * exp(beta @ X)

    Uses Breslow's method for handling tied event times and estimates
    the baseline hazard for survival function prediction.
    """

    def __init__(
        self,
        num_bins: int,
        alpha: float = 0.0,
        ties: str = 'breslow',
        n_iter: int = 100,
        tol: float = 1e-9,
        window_sizes: List[int] = None,
        feature_names: List[str] = None,
        **kwargs
    ):
        """
        Initialize Cox Proportional Hazards model.

        Args:
            num_bins: Number of discrete time bins
            alpha: Regularization parameter (0 = no regularization)
            ties: Method for handling tied event times ('breslow' or 'efron')
            n_iter: Maximum number of iterations for optimization
            tol: Convergence tolerance
            window_sizes: Window sizes for rolling features
            feature_names: Names of input features
        """
        super().__init__(
            num_bins=num_bins,
            window_sizes=window_sizes,
            feature_names=feature_names
        )

        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol

        self._time_bins = None
        self._imputer = None
        self._init_model()

    def _init_model(self):
        """Initialize the Cox PH model."""
        self.model = CoxPHSurvivalAnalysis(
            alpha=self.alpha,
            ties=self.ties,
            n_iter=self.n_iter,
            tol=self.tol
        )
        self._imputer = SimpleImputer(strategy='median')
        self._scaler = StandardScaler()  # Standardize features to prevent overflow
        self._not_all_nan_mask = None  # Mask for non-all-NaN columns
        self._not_constant_mask = None  # Mask for non-constant columns (applied after imputation)
        self._not_correlated_mask = None  # Mask for non-highly-correlated columns

    def _fit_model(self, train_data: Dict) -> Dict:
        """
        Fit the Cox Proportional Hazards model.

        Args:
            train_data: Dictionary with keys 'features', 'time_to_event_bin', 'event_indicator'

        Returns:
            Training history dictionary
        """
        print('Fitting CoxPH model...')
        features = train_data['features']
        times = train_data['time_to_event_bin']
        events = train_data['event_indicator']

        # Step 1: Identify columns that are not entirely NaN
        self._not_all_nan_mask = ~np.all(np.isnan(features), axis=0)
        n_all_nan = (~self._not_all_nan_mask).sum()
        if n_all_nan > 0:
            print(f'Dropping {n_all_nan} all-NaN feature columns')
        features_clean = features[:, self._not_all_nan_mask]

        # Step 2: Impute remaining NaNs (imputer is fit on this intermediate shape)
        features_clean = self._imputer.fit_transform(features_clean)

        # Step 3: Drop constant (zero variance) columns
        variances = np.var(features_clean, axis=0)
        self._not_constant_mask = variances > 1e-10
        n_constant = (~self._not_constant_mask).sum()
        if n_constant > 0:
            print(f'Dropping {n_constant} constant feature columns')
        features_clean = features_clean[:, self._not_constant_mask]

        # Step 4: Remove highly correlated features to reduce multicollinearity
        corr_matrix = np.abs(np.corrcoef(features_clean, rowvar=False))
        np.fill_diagonal(corr_matrix, 0)
        corr_threshold = 0.95
        to_drop = set()
        for i in range(corr_matrix.shape[0]):
            if i in to_drop:
                continue
            for j in range(i + 1, corr_matrix.shape[1]):
                if j not in to_drop and corr_matrix[i, j] > corr_threshold:
                    to_drop.add(j)
        if to_drop:
            print(f'Dropping {len(to_drop)} highly correlated features (r>{corr_threshold})')
        self._not_correlated_mask = np.ones(features_clean.shape[1], dtype=bool)
        self._not_correlated_mask[list(to_drop)] = False
        features_clean = features_clean[:, self._not_correlated_mask]

        # Step 5: Standardize features to prevent numerical overflow in exp(beta @ X)
        features_clean = self._scaler.fit_transform(features_clean)

        print(f'Final feature count: {features_clean.shape[1]} (from {features.shape[1]})')

        # Create structured array for scikit-survival
        y_structured = np.array(
            [(bool(e), t) for e, t in zip(events, times)],
            dtype=[('event', bool), ('time', float)]
        )

        # Fit model with automatic regularization escalation on numerical issues
        fitted = False
        alpha_try = self.alpha
        for attempt in range(3):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.model = CoxPHSurvivalAnalysis(
                        alpha=alpha_try,
                        ties=self.ties,
                        n_iter=self.n_iter,
                        tol=self.tol
                    )
                    self.model.fit(features_clean, y_structured)
                ill_cond = any('Ill-conditioned' in str(w.message) or 'LinAlgWarning' in str(w.category) for w in caught)
                if ill_cond and attempt < 2:
                    alpha_try = max(alpha_try * 10, 0.1) if alpha_try > 0 else 0.1
                    print(f'Ill-conditioned matrix detected, retrying with alpha={alpha_try}')
                    continue
                fitted = True
                break
            except np.linalg.LinAlgError:
                alpha_try = max(alpha_try * 10, 0.1) if alpha_try > 0 else 0.1
                print(f'Singular matrix encountered, retrying with alpha={alpha_try}')

        if not fitted:
            # Final fallback with strong regularization
            print(f'Fitting with strong regularization alpha={alpha_try}')
            self.model = CoxPHSurvivalAnalysis(
                alpha=alpha_try,
                ties=self.ties,
                n_iter=self.n_iter,
                tol=self.tol
            )
            self.model.fit(features_clean, y_structured)
        self._is_fitted = True

        # Store time bins
        self._time_bins = np.arange(self.num_bins)

        return {
            'n_samples': len(times),
            'n_events': int(events.sum()),
            'n_censored': int(len(events) - events.sum()),
            'n_features_original': features.shape[1],
            'n_features_final': features_clean.shape[1],
            'coefficients': self.model.coef_.tolist()
        }

    def _predict_pmfs(self, input_features: np.ndarray) -> np.ndarray:
        """
        Predict PMFs from model.

        Args:
            input_features: Tabular features (n_samples, n_features)

        Returns:
            PMF predictions (n_samples, num_bins)
        """
        # Step 1: Remove all-NaN columns (same as training)
        if self._not_all_nan_mask is not None:
            input_features = input_features[:, self._not_all_nan_mask]

        # Step 2: Impute NaNs (imputer was fit on this shape)
        input_features = self._imputer.transform(input_features)

        # Step 3: Remove constant columns (same as training)
        if self._not_constant_mask is not None:
            input_features = input_features[:, self._not_constant_mask]

        # Step 4: Remove highly correlated features (same as training)
        if self._not_correlated_mask is not None:
            input_features = input_features[:, self._not_correlated_mask]

        # Step 5: Standardize features (same as training)
        input_features = self._scaler.transform(input_features)

        survival_funcs = self.model.predict_survival_function(input_features, return_array=True)
        print(input_features.shape, survival_funcs.shape)

        # P(T=t) = S(t-1) - S(t)
        pmf = survival_funcs[:, :-1] - survival_funcs[:, 1:]
        pmf = np.concatenate((1 - survival_funcs[:, 0][:, np.newaxis], pmf), axis=1)
        return pmf  # (n_samples, num_bins)

    def get_coefficients(self) -> Dict[str, float]:
        """Get model coefficients (log hazard ratios)."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = self.get_feature_names()
        # Filter by not_all_nan mask first
        if self._not_all_nan_mask is not None:
            feature_names = [f for f, keep in zip(feature_names, self._not_all_nan_mask) if keep]
        # Then filter by not_constant mask
        if self._not_constant_mask is not None:
            feature_names = [f for f, keep in zip(feature_names, self._not_constant_mask) if keep]
        # Then filter by not_correlated mask
        if self._not_correlated_mask is not None:
            feature_names = [f for f, keep in zip(feature_names, self._not_correlated_mask) if keep]
        coefficients = self.model.coef_

        return dict(zip(feature_names, coefficients))

    def get_hazard_ratios(self) -> Dict[str, float]:
        """Get hazard ratios (exp of coefficients)."""
        coeffs = self.get_coefficients()
        return {k: np.exp(v) for k, v in coeffs.items()}
