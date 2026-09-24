"""Estimate a dependence structure from real background trajectories, and
condition on it — the data-driven counterpart of
``ground_truth.generate_samples._build_observed_precision``/``_coalition_value``.

Fit once, globally, from a background pool; reuse the fitted ``ObservedPrecision``
across every sample explained with :class:`~..kernel_shap.ObservationalBackground`.
No importance-reweighting branch here (unlike the ground truth): a VAR fit *is*
the working model for real data, there is no known "true" nonlinear density to
correct a linearised proposal toward.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy import sparse as _sparse

from ..precision import ObservedPrecision, condition_and_sample


@dataclass
class VARFit:
    feature_names: list[str]  # K names, in the order A's axes use
    p: int  # lag order
    A: np.ndarray  # (p, K, K); A[lag-1][target_f, source_g]
    intercept: np.ndarray  # (K,)
    sigma_eps: np.ndarray  # (K, K) residual covariance
    stationary_mean: np.ndarray  # (K,), solves mean = sum(A)@mean + intercept


def fit_var(pool: np.ndarray,
           feature_names: list[str],
           max_lag: int = 5,
           ridge: float = 1e-6) -> VARFit:
    """Fit one global VAR(p) by pooled OLS over every fully-observed lag window
    across every trajectory in ``pool`` — a single shared dependence structure,
    not one per trajectory. ``pool`` is (N, T, K), NaN where a trajectory
    doesn't cover a frame (late entry / death); windows touching NaN are
    dropped. ``p`` is chosen by (an AIC-style) minimising ``T*log|Sigma_eps| +
    2*num_params`` over ``1..max_lag``.
    """
    N, T, K = pool.shape
    # Empirical mean, not the theoretical VAR stationary mean (I -
    # sum(A))^-1 @ intercept: that formula divides by something close to
    # singular whenever the fitted process is near a unit root (ar_coeff
    # close to 1, as several of these scenarios deliberately use), where a
    # finite stationary mean barely exists in the first place. A near-zero
    # determinant there blows the solved mean up to whatever huge value
    # floating-point noise in (I - sum(A)) happens to produce — e.g. this
    # gave mean_vec entries of 15-17 for features whose real values are
    # small, which then dominates every free (masked-out) node's fill value
    # and one feature's SHAP score along with it. The pool's own empirical
    # mean is bounded, data-grounded, and doesn't depend on inverting
    # anything.
    mean = np.nanmean(pool, axis=(0, 1))
    best = None
    for p in range(1, max_lag + 1):
        X_rows, Y_rows = [], []
        for n in range(N):
            traj = pool[n]
            for t in range(p, T):
                window = traj[t - p:t + 1]  # (p+1, K), oldest first
                if np.isnan(window).any():
                    continue
                lags = window[:-1][::-1]  # (p, K): lag1, lag2, ..., lag p
                X_rows.append(lags.reshape(-1))
                Y_rows.append(window[-1])
        if len(X_rows) < K * p + K + 1:
            continue  # not enough fully-observed windows to fit this lag order
        X = np.asarray(X_rows)  # (n_windows, K*p)
        Y = np.asarray(Y_rows)  # (n_windows, K)
        X_design = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        coef, *_ = np.linalg.lstsq(X_design, Y, rcond=None)  # (K*p+1, K)
        A_flat, intercept = coef[:-1], coef[-1]
        resid = Y - X_design @ coef
        sigma_eps = np.cov(resid, rowvar=False) + ridge * np.eye(K)

        score = X.shape[0] * np.log(np.linalg.det(sigma_eps)) + 2 * (K * K * p)
        if best is None or score < best[0]:
            # A_flat block for lag l is (source_g, target_f)-indexed; transpose
            # each block to the (target_f, source_g) convention used below.
            A = np.stack(
                [A_flat[lag * K:(lag + 1) * K].T for lag in range(p)])
            best = (score, VARFit(feature_names, p, A, intercept, sigma_eps,
                                  mean))
    if best is None:
        raise ValueError(
            'Not enough fully-observed windows in the background pool to fit '
            'a VAR model at any lag order up to max_lag.')
    return best[1]


def build_precision(var_fit: VARFit, lf: int, ridge: float = 1e-8,
                    tol: float = 1e-12) -> 'VARPrecision':
    """Precision over (feature, frame) nodes for a window of length ``lf``,
    from a fitted VAR. Mirrors ``_build_observed_precision``'s per-node
    innovation-equation construction, generalised in two ways: coefficients
    come from ``var_fit.A`` instead of rule Jacobians, and equations are
    weighted by the full residual covariance ``Sigma_eps`` (features' shocks
    at the same frame are generally correlated for real data) instead of an
    independent per-feature scalar variance.
    """
    K, p = len(var_fit.feature_names), var_fit.p
    feat_idx = {f: i for i, f in enumerate(var_fit.feature_names)}
    # Node columns feature-major (matches SampleWithSHAP's (K, lf) layout);
    # equation rows time-major (makes the residual-covariance weighting a
    # plain block-diagonal matrix, one Sigma_eps block per frame).
    col = {(f, t): fi * lf + t for f, fi in feat_idx.items() for t in range(lf)}
    n_obs = K * lf

    rows, cols, vals = [], [], []
    eq = 0
    for t in range(lf):
        for f, fi in feat_idx.items():
            rows.append(eq)
            cols.append(col[(f, t)])
            vals.append(1.0)
            if t >= p:
                for lag in range(1, p + 1):
                    for g, gi in feat_idx.items():
                        c = var_fit.A[lag - 1][fi, gi]
                        if abs(c) > tol:
                            rows.append(eq)
                            cols.append(col[(g, t - lag)])
                            vals.append(-float(c))
            eq += 1
    incidence = _sparse.csc_matrix((vals, (rows, cols)), shape=(eq, n_obs))

    sigma_inv = np.linalg.inv(var_fit.sigma_eps)
    L_prec = np.linalg.cholesky(sigma_inv)  # L_prec @ L_prec.T = sigma_inv
    D_block = _sparse.block_diag([sigma_inv] * lf).tocsc()
    D_sqrt_block = _sparse.block_diag([L_prec] * lf).tocsc()

    Q = (incidence.T @ D_block @ incidence + ridge * _sparse.eye(n_obs)).tocsc()

    mean_vec = np.empty(n_obs)
    for f, fi in feat_idx.items():
        for t in range(lf):
            mean_vec[col[(f, t)]] = var_fit.stationary_mean[fi]

    prec = ObservedPrecision(n_obs, incidence, D_sqrt_block, Q, mean_vec)
    return VARPrecision(var_fit.feature_names, lf, prec)


@dataclass
class VARPrecision:
    """A shared ``ObservedPrecision`` plus the (feature, frame) shape metadata
    needed to reshape flat node vectors back to ``(K, lf)`` — metadata the
    shared type doesn't carry since ground truth's own indexing works the same
    way but isn't tied to a fixed ``lf`` the way a VAR window is."""
    feature_names: list[str]
    lf: int
    prec: ObservedPrecision


def sample_conditional(var_prec: VARPrecision, pinned_mask: np.ndarray,
                       basevec: np.ndarray, num_samples: int,
                       xi: np.ndarray | None = None) -> np.ndarray:
    """Sample the free (feature, frame) nodes given the pinned ones.

    ``pinned_mask``/``basevec`` are (n_obs,), feature-major node order
    (``col[(f,t)] = fi*lf+t``, matching ``build_precision``). Returns
    ``(K, lf, num_samples)`` — pinned entries equal ``basevec`` in every
    sample, free entries are conditional draws given the pinned ones. Thin
    wrapper around the shared perturb-and-solve core in ``..precision``; VAR
    has no importance-reweighting step (there's no "true" nonlinear density to
    correct toward — the fit *is* the working model), so only ``.values`` is
    used from the shared result.

    ``xi``: (n_eq, num_samples) standard normal noise; drawn fresh if omitted.
    A caller evaluating several coalitions for the *same* Shapley explanation
    (e.g. ``ObservationalBackground``) should pass the same ``xi`` to every
    one of them — common random numbers, so a player the precision treats as
    independent of everything else gets an exactly-zero marginal contribution
    instead of noise from differencing independently-sampled draws (mirrors
    ``ground_truth.generate_samples._make_obs_value_factory``, which redraws
    its own CRN seed once per permutation for the same reason).
    """
    prec = var_prec.prec
    if xi is None:
        xi = np.random.normal(size=(prec.incidence.shape[0], num_samples))
    result = condition_and_sample(prec, pinned_mask, basevec, xi)
    K = len(var_prec.feature_names)
    return result.values.reshape(K, var_prec.lf, num_samples)
