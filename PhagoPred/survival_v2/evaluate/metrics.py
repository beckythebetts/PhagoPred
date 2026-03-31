from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score as sk_integrated_brier_score
from sksurv.metrics import brier_score as sk_brier_score

# def concordance_index(predicted_pmf: np.ndarray, true_times: np.ndarray,
#                       event_indicators: np.ndarray, bin_edges: np.ndarray):
#     """
#     Compute concordance index (C-index) for survival predictions at each time bin
#     using sksurv.

#     Args
#     ----
#         predicted_pmf: (n, num_bins) predicted probability mass function
#         true_times: (n,) observed times
#         event_indicators: (n,) 1 if event, 0 if censored
#     Returns
#     -------
#         c_index: float in [0, 1], 0.5 is random
#     """
#     bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#     # predicted_times = np.mean(predicted_pmf * bin_centres, axis=1)
#     pmf_sum = predicted_pmf.sum(axis=1, keepdims=True).clip(min=1e-8)
#     predicted_times = (predicted_)
#     risk = -predicted_times

#     true_times = np.array(true_times)
#     event_indicators = np.array(event_indicators, dtype=bool)

#     c_index = concordance_index_censored(event_indicators, true_times, risk)[0]
#     return c_index


def concordance_index(predicted_pmf: np.ndarray, true_times: np.ndarray,
                      event_indicators: np.ndarray,
                      bin_edges: np.ndarray) -> np.ndarray:
    """Calculate concordance index at each time bin using CIF as risk score.
    Return
    ------
        CIndex per bin: [num_bins, ]"""
    cif = np.cumsum(predicted_pmf, axis=1)
    c_index_per_bin = np.array([
        concordance_index_censored(event_indicators.astype(bool), true_times,
                                   cif[:, t])[0] for t in range(cif.shape[1])
    ])
    return c_index_per_bin


def reciever_operator_characteristic(
        event_probabilities: np.ndarray,
        events: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute time-dependent AUC for survival predictions.
    Binary equivalent to concordance index.
    
    Args
    ----
        event_probabilities: (n,) predicted probability of event by a certain time
        events: (n,) 1 if event occurred, 0 if censored
    Returns
    -------
        auc: float, area under the ROC curve
        fpr: (m,) false positive rates (for plotting roc curve)
        tpr: (m,) true positive rates (for plotting roc curve)
        thresholds: (m,)
    """
    fpr, tpr, thresholds = roc_curve(events, event_probabilities)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr, thresholds


def integrated_brier_score(
        pmf: np.ndarray, true_times: np.ndarray, event_indicators: np.ndarray,
        bin_edges: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute integrated Brier score for discrete time survival predictions.
    Using sksurv.

    Args
    ----
        pmf: (n, num_bins) predicted PMF 
        true_times: (n,) true event times
        event_indicators: (n,) 1 if event, 0 if censored
        bin_edges: (num_bins + 1,) edges of the time bins
        
    Returns
    -------
        ibs: float, integrated Brier score
        bs: (num_bins,) Brier score at each time point (for plotting)
        times: (num_bins,) time points corresponding to Brier scores (for plotting)
    """
    survival_train = np.array([(e, t)
                               for e, t in zip(event_indicators, true_times)],
                              dtype=[('event', bool), ('time', float)])
    survival_test = survival_train

    cif = pmf.cumsum(axis=1)
    survival_probs = 1.0 - cif

    min_time = survival_test["time"].min()
    max_time = survival_test["time"].max()

    times = bin_edges[1:]
    valid = (times > min_time) & (times < max_time)

    times = times[valid]
    survival_probs = survival_probs[:, valid]

    times_out, bs = sk_brier_score(survival_train, survival_test,
                                   survival_probs, times)
    ibs = sk_integrated_brier_score(survival_train, survival_test,
                                    survival_probs, times)

    return ibs, times_out, bs


def mean_squared_error(event_probabilities: np.ndarray,
                       events: np.ndarray) -> float:
    """
    Compute mean squared error (binary equivalent to Brier score at a single time point).

    Args
    ----
        event_probabilities: (n,) predicted probability of event by a certain time
        events: (n,) 1 if event occurred, 0 if censored
    Returns
    -------
        mse: float, mean squared error
    """
    mse = np.mean((event_probabilities - events)**2)
    return mse


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence D_KL(P || Q) for discrete distributions.

    Args:
        p: (batch_size, num_bins) true distribution
        q: (batch_size, num_bins) predicted distribution
    Returns:
        kl_div: (batch_size,) KL divergence for each sample
    """
    if isinstance(p, np.ndarray):
        p = torch.from_numpy(p)
    if isinstance(q, np.ndarray):
        q = torch.from_numpy(q)
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10
    kl_div = torch.sum(p * torch.log(p / q), dim=1)
    return torch.mean(kl_div, dim=0).cpu().numpy()


def binary_cross_entropy(predicted_probs: np.ndarray,
                         true_labels: np.ndarray) -> float:
    """
    Compute binary cross-entropy loss. (Equivalent to kl-divergence for binar case)

    Args:
        predicted_probs: (n,) predicted probabilities of the positive class
        true_labels: (n,) true binary labels (0 or 1)   
    Returns:
        bce: float, binary cross-entropy loss
    """
    bce = -np.mean(true_labels * np.log(predicted_probs + 1e-10) +
                   (1 - true_labels) * np.log(1 - predicted_probs + 1e-10))
    return bce
