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


def node_noise(prec: ObservedPrecision, xi: np.ndarray) -> np.ndarray:
    """incidenceᵀ D_sqrt xi, (n_obs, B). For any free set F, its rows F are
    ~ N(0, Q_FF): ``incidence[:, F].T @ v == (incidence.T @ v)[F]``, so this
    is computed once per xi and indexed per coalition."""
    return np.asarray(prec.incidence.T @ (prec.D_sqrt @ xi))


def sample_free(prec: ObservedPrecision,
                Qcsr,
                free: np.ndarray,
                pinned: np.ndarray,
                basevec: np.ndarray,
                noise: np.ndarray) -> tuple[np.ndarray, object, object,
                                            np.ndarray]:
    """Conditional draws of the free nodes given the pinned ones at
    ``basevec``: (len(free), B), plus (lu, Q_FF, rhs).

    ``Qcsr`` is ``prec.Q`` as CSR and ``noise`` is ``node_noise(prec, xi)``,
    both computed once and reused across coalitions."""
    QF = Qcsr[free]
    Qff = QF[:, free].tocsc()
    lu = _splu(Qff)
    if len(pinned):
        rhs = -(QF[:, pinned] @ (basevec[pinned] - prec.mean_vec[pinned]))
    else:
        rhs = np.zeros(len(free))
    Z = lu.solve(rhs[:, None] + noise[free])  # deviations from the mean
    return Z + prec.mean_vec[free, None], lu, Qff, rhs


def condition_and_sample(prec: ObservedPrecision, pinned_mask: np.ndarray,
                         basevec: np.ndarray,
                         xi: np.ndarray) -> ConditionResult:
    """All nodes (n_obs, B): pinned at ``basevec``, free drawn conditionally.
    For many coalitions with the same xi, precompute ``node_noise`` and the
    CSR matrix and call ``sample_free`` directly."""
    F = np.where(~pinned_mask)[0]
    P = np.where(pinned_mask)[0]
    B = xi.shape[1]
    values = np.repeat(basevec[:, None], B, axis=1)
    if len(F) == 0:
        return ConditionResult(values, None, None, None, F, P)
    values[F], lu, Qff, rhs = sample_free(prec, prec.Q.tocsr(), F, P,
                                          basevec, node_noise(prec, xi))
    return ConditionResult(values, lu, Qff, rhs, F, P)
