"""
Vector autoregression fit.

A VAR(p) fitted to background histories: the data-driven dependence
structure for ``ObservationalVAR``, the counterpart of the graph's own
``linearised_precision`` used by ``ObservationalGT``.

The model (``A``, ``sigma_eps``, ``mean``) is fitted once and shared; the
precision over a window's (feature, frame) nodes is built from it per window
length.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from ..utils.precision import ObservedPrecision
from .backgrounds import BackgroundSet

# p = max lag
# K = num features


@dataclass
class VARFit:
    feature_names: list[str]  # K, the order of A's axes
    lags: tuple[int, ...]
    A: np.ndarray  # (len(lags), K, K); A[lag - 1][target, source]
    intercept: np.ndarray  # (K,)
    sigma_eps: np.ndarray  # (K, K) residual covariance
    mean: np.ndarray  # (K,) empirical mean of the fitted rows
    # The criterion (AIC or BIC) for orders 1..max_lag (index p - 1), all
    # scored on the same windows; p is its argmin.
    order_scores: np.ndarray | None = None

    def companion_radius(self) -> float:
        """Spectral radius of the companion matrix (>= 1: non-stationary).
        Companion matrix stacks VAR with p lags into one VAR, with state vector of slast p frames stacked.
        Comapanion radius is how fast the VAR fir "forgets", determing how long the burn in needs to be.
        
        Companion matrix like (each enery is K x K matrix)
                x_{t-1}    x_{t-2}    x_{t-3}
        x_t    [  A₁    |    A₂    |    A₃   ]
        x_{t-1}[  I     |    0     |    0    ]    
        x_{t-2}[  0     |    I     |    0    ]
        
        First row is coefficients, other rows (identity) just allow the matrix to be applied repatably to get decay rate.
        """
        K, p = len(self.feature_names), self.p
        companion = np.zeros((K * p, K * p))
        companion[:K] = np.concatenate([self.coef(l) for l in range(1, p + 1)],
                                       axis=1)
        companion[K:, :-K] = np.eye(K * (p - 1))
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    def coef(self, lag: int) -> np.ndarray:
        if lag in self.lags:
            return self.A[self.lags.index(lag)]
        return np.zeros((len(self.feature_names), ) * 2)

    @property
    def p(self) -> int:
        return max(self.lags)


def fit_var(bg_set: BackgroundSet,
            feature_names: list[str],
            max_lag: int = 100,
            criterion: Literal['aic', 'bic'] = 'aic',
            ridge: float = 1e-6) -> VARFit:
    """One global VAR(p), lags 1..p, by pooled OLS over every NaN-free
    (p + 1)-frame window of every background row.

    ``p`` in ``1..max_lag`` is chosen by AIC (or BIC), every p scored on the
    same windows of ``max_lag + 1`` frames (``select_order``); the chosen p
    is then refitted on all of its own windows. Scoring each p on its own
    windows instead biases the choice by the number of windows, not the fit.
    AIC rather than BIC: with a sharp long lag (B reading A 50 frames back),
    BIC's log(n) penalty on the empty lags in between outweighs it.

    The mean is the empirical one, not (I - sum A)^-1 c: near a unit root
    (AR ~0.99, as the presets use) that inverse is close to singular and
    blows up."""
    cols = [bg_set.feature_names.index(n) for n in feature_names]
    rows = [h[:, cols] for h in bg_set.vals]
    mean = np.nanmean(np.concatenate(rows, axis=0), axis=0)

    scores = select_order(rows, mean, max_lag, criterion)
    p = int(np.argmin(scores)) + 1
    fit = _ols(rows, p, ridge)
    if fit is None:
        raise ValueError(f'not enough NaN-free windows to fit VAR({p})')
    A, intercept, sigma_eps, _ = fit
    return VARFit(list(feature_names), tuple(range(1, p + 1)), A, intercept,
                  sigma_eps, mean, scores)


def _ols(rows: list[np.ndarray], p: int, ridge: float):
    """OLS (oridnary least squares) of x_t on x_{t-1..t-p} and a constant, over every NaN-free window
    of p + 1 frames. Returns (A (p, K, K), intercept (K,), sigma_eps (K, K),
    n windows), or None with too few windows."""
    K = rows[0].shape[1]
    X, Y = _lagged_design(rows, tuple(range(1, p + 1)))
    if len(X) < K * p + K + 1:
        return None
    design = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)  # (K p + 1, K)
    resid = Y - design @ coef
    sigma_eps = np.cov(resid, rowvar=False) + ridge * np.eye(K)
    # coef block for lag l is [source, target]; store [target, source].
    A = np.stack([coef[l * K:(l + 1) * K].T for l in range(p)])
    return A, coef[-1], sigma_eps, len(X)


def select_order(rows: list[np.ndarray], mean: np.ndarray, max_lag: int,
                 criterion: Literal['aic', 'bic'] = 'aic') -> np.ndarray:
    """AIC or BIC of the VAR(p), p = 1..max_lag, all on the same targets:
    every NaN-free window of max_lag + 1 frames (scores are only comparable
    between models of the same data; cells shorter than that don't count
    here). Returns the score by order, index p - 1.

        score = n log det(sigma) + penalty * K (K p + 1)

    with penalty 2 (AIC) or log(n) (BIC) per coefficient.

    OLS needs only X'X, X'Y and Y'Y, so these are accumulated once, cell by
    cell, with all max_lag lags and a constant as columns (never holding the
    full design matrix: n ~ 300k rows by K max_lag + 1 columns). The columns
    are lag-major, so VAR(p) uses the first K p of them plus the constant,
    and each order costs one small solve:

        B = (X_S'X_S)^-1 X_S'Y,   sigma = (Y'Y - B'X_S'Y) / n

    Data are centred on ``mean`` first: with a constant column, uncentred
    near-unit-root signals make X'X badly conditioned."""
    if criterion not in ('aic', 'bic'):
        raise ValueError(f'unknown criterion {criterion!r}')
    K = rows[0].shape[1]
    d = K * max_lag + 1
    XtX, XtY = np.zeros((d, d)), np.zeros((d, K))
    YtY, n = np.zeros((K, K)), 0
    all_lags = tuple(range(1, max_lag + 1))
    for h in rows:
        X, Y = _lagged_design([h - mean], all_lags)
        if not len(X):
            continue
        X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        XtX += X.T @ X
        XtY += X.T @ Y
        YtY += Y.T @ Y
        n += len(X)
    if n < d:
        raise ValueError(f'only {n} NaN-free windows of {max_lag + 1} frames: '
                         f'too few to compare VAR orders up to {max_lag}')

    penalty = 2.0 if criterion == 'aic' else np.log(n)
    scores = np.empty(max_lag)
    for p in range(1, max_lag + 1):
        S = np.r_[0:K * p, d - 1]  # lags 1..p, then the constant
        B = np.linalg.lstsq(XtX[np.ix_(S, S)], XtY[S], rcond=None)[0]
        sigma = (YtY - B.T @ XtY[S]) / n
        sigma = (sigma + sigma.T) / 2  # symmetric up to rounding
        scores[p - 1] = (n * np.linalg.slogdet(sigma)[1] + penalty * K *
                         (K * p + 1))
    return scores


def _lagged_design(rows: list[np.ndarray],
                   lags: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """X (n, len(lags) K): x_{t-l} for each l in ``lags``, lag-major (in the
    order of ``lags``); Y (n, K): x_t. One row per NaN-free window of
    max(lags) + 1 frames, so every row has all its lags.

    ``lags = range(1, p + 1)`` gives the contiguous VAR(p) design."""
    p = max(lags)
    # After reversing a window's history, index l - 1 is lag l.
    lag_idx = [l - 1 for l in lags]
    Xs, Ys = [], []
    for h in rows:
        if len(h) <= p:
            continue
        w = sliding_window_view(h, p + 1, axis=0)  # (T - p, K, p + 1)
        w = w[~np.isnan(w).any(axis=(1, 2))]
        history = w[..., :-1][..., ::-1]  # (n, K, p): index l - 1 is lag l
        Xs.append(history[..., lag_idx].transpose(0, 2, 1).reshape(len(w), -1))
        Ys.append(w[..., -1])
    K = rows[0].shape[1]
    if not Xs:
        return np.empty((0, K * len(lags))), np.empty((0, K))
    return np.concatenate(Xs), np.concatenate(Ys)


def default_burn_in(var_fit: VARFit,
                    tol: float = 0.05,
                    max_burn_in: int = 1000) -> int:
    """Free context frames needed for the VAR's marginal variance to reach
    within ``tol`` of stationary (the gap decays like rho^(2t)). Without them
    the first frames' prior is one innovation around the mean, far tighter
    than the data, and revealing them earns spurious importance.

    The first ``p`` frames have no VAR equation (``build_precision``), so
    the variance only starts building after them: they are added on top."""
    rho = var_fit.companion_radius()
    if rho >= 1.0:
        return max_burn_in
    decay = int(np.ceil(np.log(tol) / (2 * np.log(rho))))
    return min(max_burn_in, decay + var_fit.p)


def build_precision(var_fit: VARFit,
                    num_frames: int,
                    ridge: float = 1e-8,
                    tol: float = 1e-12) -> ObservedPrecision:
    """Precision over the (feature, frame) nodes of a ``num_frames`` window
    (context included), feature-major: node ``f * num_frames + t``, matching
    ``Observational._flatten``.

    One innovation equation per (frame, feature),
    ``x_t - sum_l A_l x_{t-l} = eps_t``, weighted by the full ``sigma_eps``
    (shocks of different features at the same frame can be correlated).
    Equations are in deviations from ``mean``; that drops the intercept,
    exact when ``mean`` is the stationary mean."""
    K, p = len(var_fit.feature_names), var_fit.p
    T = num_frames
    n_obs = K * T

    rows, cols, vals = [], [], []
    eq = 0
    for t in range(T):  # equations time-major: Sigma weighting block-diagonal
        for f in range(K):
            rows.append(eq)
            cols.append(f * T + t)
            vals.append(1.0)
            if t >= p:
                for i, lag in enumerate(var_fit.lags):
                    # for lag in range(1, p + 1):
                    for k in range(K):
                        c = var_fit.A[i][f, k]
                        if abs(c) > tol:
                            rows.append(eq)
                            cols.append(k * T + t - lag)
                            vals.append(-float(c))
            eq += 1
    incidence = sparse.csc_matrix((vals, (rows, cols)), shape=(eq, n_obs))

    sigma_inv = np.linalg.inv(var_fit.sigma_eps)
    # D_sqrt D_sqrtᵀ = sigma_inv, so incidenceᵀ D_sqrt xi ~ N(0, Q)
    L_prec = np.linalg.cholesky(sigma_inv)  # L Lᵀ = sigma_inv
    D_sqrt = sparse.block_diag([L_prec] * T).tocsc()
    D = sparse.block_diag([sigma_inv] * T).tocsc()
    Q = (incidence.T @ D @ incidence + ridge * sparse.eye(n_obs)).tocsc()

    mean_vec = np.repeat(var_fit.mean, T)  # feature-major
    return ObservedPrecision(n_obs, incidence, D_sqrt, Q, mean_vec)
