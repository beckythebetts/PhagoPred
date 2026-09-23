"""Shared precision-matrix conditioning core.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu as _splu


@dataclass
class ObservedPrecision:
    n_obs: int  # number of (feature, frame) nodes
    incidence: object  # scipy csc (n_eq, n_obs): one linearised eps-equation per row
    D_sqrt: object  # scipy sparse (n_eq, n_eq): sqrt of the per-equation precision
    #   weighting. Diagonal for independent per-feature noise (ground truth);
    #   block-diagonal for correlated cross-feature residuals (VAR). Q's
    #   "structural square root" for sampling is incidenceᵀ @ D_sqrt.
    Q: object  # scipy csc (n_obs, n_obs) = incidenceᵀ (D_sqrt D_sqrtᵀ) incidence + ridge
    mean_vec: np.ndarray  # (n_obs,): zero for an implicitly zero-mean process
    #   (ground truth); the stationary mean broadcast to every node otherwise.


@dataclass
class ConditionResult:
    values: np.ndarray  # (n_obs, B): pinned rows = basevec; free rows = samples
    lu: object | None  # splu(Q_FF), or None if nothing was free (F empty)
    Qff: object | None
    rhs: np.ndarray | None  # Q_FF·mean = rhs; a reweighting caller wants
    #   ``mean = lu.solve(rhs)`` without redoing this from scratch.
    F: np.ndarray  # free node indices
    P: np.ndarray  # pinned node indices


def condition_and_sample(prec: ObservedPrecision, pinned_mask: np.ndarray,
                         basevec: np.ndarray,
                         xi: np.ndarray) -> ConditionResult:
    """Perturb-and-solve conditional sampling — the core shared by every
    precision-based estimator.

    ``xi``: (n_eq, B) standard normal noise; B columns become B conditional
    draws. Pinned rows of ``values`` equal ``basevec`` in every column; free
    rows are draws from the conditional Gaussian given the pinned ones, with
    all conditioning done in deviation-from-``mean_vec`` space (the innovation
    equations are only zero-mean there) and the mean added back before return.

    Returns the LU factor and ``Q_FF`` alongside the samples so a caller that
    needs an importance-reweighting correction (e.g. ground truth's nonlinear
    rules, where ``Q`` is only a local linearisation) can build it from the
    same solve without redoing the factorisation — a caller with nothing to
    reweight (a VAR fit, which *is* the working model) can just ignore them.
    """
    F = np.where(~pinned_mask)[0]
    P = np.where(pinned_mask)[0]
    B = xi.shape[1]
    values = np.repeat(basevec[:, None], B, axis=1)
    if len(F) == 0:
        return ConditionResult(values, None, None, None, F, P)

    Qcsr = prec.Q.tocsr()
    Qff = Qcsr[F][:, F].tocsc()
    lu = _splu(Qff)
    x_dev_P = basevec[P] - prec.mean_vec[P]
    rhs = -(Qcsr[F][:, P] @ x_dev_P) if len(P) else np.zeros(len(F))
    w = prec.incidence[:, F].T @ (prec.D_sqrt @ xi)  # ~ N(0, Q_FF)
    Z = lu.solve(rhs[:, None] + w)  # deviation-space conditional samples

    values[F] = Z + prec.mean_vec[F, None]
    return ConditionResult(values, lu, Qff, rhs, F, P)
