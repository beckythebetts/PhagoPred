from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
"""
M = num_players
R = num groups (Each group is a subset of the bg_fill_draws)
K = R + 1  (value function returns the pooled value in column 0, then one value per group)

Z = coalitions [num_coalitions, M]
y = value function result
w = coalition weights
"""


@dataclass
class SHAPEstimate:
    values: np.ndarray  # (M, K) value_function returns
    none_pinned: np.ndarray  # (K, )
    all_pinned: np.ndarray  # (K, )
    num_evaluations: int
    exact: bool

    @property
    def shap_values(self) -> np.ndarray:
        return self.values[:, 0]

    @property
    def stderr(self) -> np.ndarray | None:
        groups = self.values[:, 1:]
        if groups.shape[1] < 2:
            return None
        return groups.std(1, ddof=1) / np.sqrt(groups.shape[1])


def kernel_shap(
    value_fn: callable,
    M: int,
    budget: int | None = None,
    rng: np.random.Generator | None = None,
) -> SHAPEstimate:
    none_pinned, all_pinned = value_fn(np.zeros(
        (1, M), bool))[0], value_fn(np.ones((1, M), bool))[0]
    total = all_pinned - none_pinned
    if M == 1:
        return SHAPEstimate(total[None], none_pinned, all_pinned, 2, True)

    (Z, w), exact = _get_coalitions(M, budget, rng)

    y = value_fn(Z) - none_pinned

    values = _solve_constrained_wls(Z, y, w, total)

    return SHAPEstimate(values=values,
                        none_pinned=none_pinned,
                        all_pinned=all_pinned,
                        num_evaluations=len(Z) + 2,
                        exact=exact)


def _get_coalitions(
    M: int,
    budget: int | None,
    rng: np.random.Generator | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], bool]:
    if budget is None:
        budget = 2 * M + 2048

    if 2**M - 2 <= budget:
        return _get_all_coalitions(M), True
    else:
        if rng is None:
            raise SHAPError(
                'Is smapling coalitions, rng must be provided for repordicbility'
            )
        return _sample_coalitions(M, budget, rng), False


def _shapley_kernel(M: int, s: np.ndarray) -> np.ndarray:
    """Weight of a coalition of size s."""
    return (M - 1) / (np.array([comb(M, int(k)) for k in s]) * s * (M - s))


def _get_all_coalitions(M: int) -> tuple[np.ndarray, np.ndarray]:
    """Only used if M is small"""
    Z = ((np.arange(1, 2**M - 1)[:, None] >> np.arange(M)) & 1).astype(bool)

    return Z, _shapley_kernel(M, Z.sum(1))


def _sample_coalitions(
        M: int, budget: int,
        rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """If M is large, avoid getting all possibel coalitiions (large meory requirement)"""
    sizes = np.arange(1, M)
    probability = (M - 1) / (sizes * (M - sizes))
    probability /= probability.sum(
    )  # Probability of drawing sample of each size when sampled with shapley_kernel
    n_pairs = budget // 2
    s = rng.choice(sizes, size=n_pairs, p=probability)
    Z = np.zeros((n_pairs, M), dtype=bool)
    for row, k in zip(Z, s):
        row[rng.choice(M, size=k, replace=False)] = True
    Z = np.concatenate([Z, ~Z])
    Z, counts = np.unique(Z, axis=0, return_counts=True)
    return Z, counts.astype(float)


def _solve_constrained_wls(Z: np.ndarray, y: np.ndarray, w: np.ndarray,
                           total: np.ndarray) -> np.ndarray:
    """
    Z [n, M] bool coalitions, y [n, K] = v(S) - v(empty), w [n] weights,
    total [K] = v(full) - v(empty). Returns [M, K] Shapley values.
    """
    if Z.shape[0] < Z.shape[1]:
        raise SHAPError(
            'Number of coalitions less than number of players, cannot solve WLS, increase budget.'
        )
    Zf = Z.astype(float)  # [n, M]
    A = Zf[:, :-1] - Zf[:, -1:]  # [n, M-1] - [n, 1] -> [n, M-1]
    b = y - Zf[:, -1:] * total  # [n, K] - ([n, 1] * [K]) -> [n, K]
    sqrt_w = np.sqrt(w)[:, None]  # [n, 1]
    shap_vals, *_ = np.linalg.lstsq(A * sqrt_w, b * sqrt_w,
                                    rcond=None)  # [M-1, K]
    shap_vals = np.vstack([shap_vals, total - shap_vals.sum(0)])  # [M, K]
    return shap_vals


class SHAPError(Exception):
    pass
