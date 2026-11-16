import numpy as np


def event_probability(hazards: np.ndarray, horizon: int) -> np.ndarray:
    """P(death in frames L+1 .. L+H | alive at L)"""
    return 1.0 - np.prod(1.0 - hazards[1:horizon + 1], axis=0)


def rmst(binned_pmf: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Restricted mean survival time up to bins[-1]. binned_pmf [n, num_bins]."""
    survival = 1.0 - np.cumsum(binned_pmf, axis=-1)
    return (survival * np.diff(bins)).sum(axis=-1)
