from __future__ import annotations
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score as sk_integrated_brier_score
from sksurv.metrics import brier_score as sk_brier_score


def hazard_mse(
    pred_pmf: np.ndarray,
    true_binned_pmfs: list[np.ndarray | None],
) -> np.ndarray | None:
    """Per-bin hazard MSE against the true underlying PMF.

    Only uses samples where the true binned PMF is available (synthetic data).
    """
    valid = [(p, t) for p, t in zip(pred_pmf, true_binned_pmfs)
             if t is not None]
    if not valid:
        return None
    pred = np.array([p for p, _ in valid])
    true = np.array([t for _, t in valid])

    true_cdf = np.cumsum(true, axis=1)
    pred_cdf = np.cumsum(pred, axis=1)
    true_sf = np.concatenate([np.ones((len(true), 1)), 1.0 - true_cdf[:, :-1]],
                             axis=1)
    pred_sf = np.concatenate([np.ones((len(pred), 1)), 1.0 - pred_cdf[:, :-1]],
                             axis=1)
    true_hazard = true / np.clip(true_sf, 1e-8, None)
    pred_hazard = pred / np.clip(pred_sf, 1e-8, None)
    return np.mean((pred_hazard - true_hazard)**2, axis=0)


def pmf_mse(pred_pmf: np.ndarray,
            true_binned_pmfs: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [(p, t) for p, t in zip(pred_pmf, true_binned_pmfs)
             if t is not None]

    pred = np.array([p for p, _ in valid])
    true = np.array([t for _, t in valid])

    return np.mean((pred - true)**2, axis=0)


def soft_confusion_matrix(pred_pmf: np.ndarray,
                          true_bins: np.ndarray) -> np.ndarray:
    final_bin_pmf = (1.0 - pred_pmf.sum(axis=1, keepdims=True))
    pmf = np.concatenate([pred_pmf, final_bin_pmf], axis=1)

    num_bins = pmf.shape[1]
    cm = np.zeros((num_bins, num_bins))
    np.add.at(cm, true_bins, pmf)

    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-8)

    return cm


# def soft_confusion_matrix(pred_pmf: np.ndarray,
#                           true_bins: np.ndarray,
#                           events: np.ndarray,
#                           num_bins: int,
#                           normalize: str | None = 'true') -> np.ndarray:
#     """Probabilistic confusion matrix that preserves the full predicted PMF.

#     Unlike the argmax/RMST confusion matrices, the prediction is never collapsed
#     to a single bin. A final bin holding the leftover mass
#     (1 - sum(pmf), i.e. the probability of surviving past the horizon) is
#     appended so the predicted axis matches the argmax CM. Row ``t`` is the summed
#     predicted distribution over the uncensored samples whose true event bin is
#     ``t``; with ``normalize='true'`` each row is the average predicted PMF for
#     those samples, so the diagonal is the mean mass placed on the correct bin.

#     Args
#     ----
#         pred_pmf: (n, num_bins) predicted PMFs.
#         true_bins: (n,) true event/censoring bin indices.
#         events: (n,) event indicators (1=event, 0=censored).
#         num_bins: number of event time bins (before the leftover bin).
#         normalize: 'true' (row), 'pred' (column) or None (raw mass counts).

#     Returns
#     -------
#         (num_bins + 1, num_bins + 1) soft confusion matrix.
#     """
#     # Only uncensored samples have a known true event bin.
#     mask = events == 1
#     leftover = (1.0 - pred_pmf.sum(axis=1, keepdims=True)).clip(min=0.0)
#     pmf = np.concatenate([pred_pmf, leftover], axis=1)[mask]
#     true = true_bins[mask].astype(int).clip(0, num_bins)

#     size = num_bins + 1
#     cm = np.zeros((size, size))
#     np.add.at(cm, true, pmf)  # cm[true_bin, :] += predicted PMF

#     if normalize == 'true':
#         cm /= cm.sum(axis=1, keepdims=True).clip(min=1e-8)
#     elif normalize == 'pred':
#         cm /= cm.sum(axis=0, keepdims=True).clip(min=1e-8)
#     return cm


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
