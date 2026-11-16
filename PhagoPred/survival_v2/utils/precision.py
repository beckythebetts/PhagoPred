from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu as _splu


@dataclass
class ObservedPrecision:
    n_obs: int  # number of (feature, frame) nodes
    incidence: object  # scipy csc (n_eq, n_obs)
    D_sqrt: object  # scipy sparse (n_eq, n_eq):
    Q: object  # scipy csc (n_obs, n_obs) = incidenceᵀ (D_sqrt D_sqrtᵀ) incidence + ridge
    mean_vec: np.ndarray  # (n_obs,)


@dataclass
class ConditionResult:
    values: np.ndarray  # (n_obs, B)
    lu: object | None  # splu(Q_FF)
    Qff: object | None
    rhs: np.ndarray | None  # Q_FF·mean
    F: np.ndarray  # free node indices
    P: np.ndarray  # pinned node indices


def condition_and_sample(prec: ObservedPrecision, pinned_mask: np.ndarray,
                         basevec: np.ndarray,
                         xi: np.ndarray) -> ConditionResult:

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
