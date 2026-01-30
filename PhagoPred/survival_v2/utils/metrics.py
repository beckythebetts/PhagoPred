from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import integrated_brier_score as sk_integrated_brier_score


# def concordance_index(predicted_times, true_times, event_indicators):
#     """
#     Compute concordance index (C-index) for survival predictions.

#     Args:
#         predicted_times: (n,) predicted survival times
#         true_times: (n,) observed times
#         event_indicators: (n,) 1 if event, 0 if censored

#     Returns:
#         c_index: float in [0, 1], 0.5 is random
#     """
#     predicted_times = np.array(predicted_times)
#     true_times = np.array(true_times)
#     event_indicators = np.array(event_indicators)

#     n = len(predicted_times)
#     concordant = 0
#     discordant = 0
#     tied = 0

#     for i in range(n):
#         if event_indicators[i] == 0:
#             continue  # Skip censored

#         for j in range(n):
#             if i == j:
#                 continue

#             # i had event, j had later time (event or censored)
#             if true_times[i] < true_times[j]:
#                 if predicted_times[i] < predicted_times[j]:
#                     concordant += 1
#                 elif predicted_times[i] > predicted_times[j]:
#                     discordant += 1
#                 else:
#                     tied += 1

#     total = concordant + discordant + tied
#     if total == 0:
#         return 0.5

#     return (concordant + 0.5 * tied) / total

def concordance_index(predicted_pmf, true_times, event_indicators, bin_edges):
    """
    Compute concordance index (C-index) for survival predictions.
    Using sksurv.

    Args:
        predicted_pmf: (n, num_bins) predicted probability mass function
        true_times: (n,) observed times
        event_indicators: (n,) 1 if event, 0 if censored
    Returns:
        c_index: float in [0, 1], 0.5 is random
    """ 
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    predicted_times = np.mean(predicted_pmf * bin_centres, axis=1)
    risk = -predicted_times
    
    true_times = np.array(true_times)
    event_indicators = np.array(event_indicators, dtype=bool)

    c_index= concordance_index_censored(
        event_indicators,
        true_times,
        risk
    )[0]
    return c_index

# def brier_score(survival_probs, time_points, true_times, event_indicators,
#                 survival_train_times=None, survival_train_events=None):
#     """
#     Compute integrated Brier score for survival predictions.

#     Args:
#         survival_probs: (n_test, n_time_points) survival probabilities at each time point
#         time_points: (n_time_points,) time points corresponding to survival_probs columns
#         true_times: (n_test,) observed times for test set
#         event_indicators: (n_test,) 1 if event, 0 if censored for test set
#         survival_train_times: (n_train,) observed times for training set (required for censoring adjustment)
#         survival_train_events: (n_train,) event indicators for training set (required for censoring adjustment)

#     Returns:
#         ibs: float integrated Brier score
#     """
#     # Convert to numpy arrays
#     survival_probs = np.array(survival_probs)
#     time_points = np.array(time_points)
#     true_times = np.array(true_times)
#     event_indicators = np.array(event_indicators, dtype=bool)

#     # Create structured arrays for sksurv
#     survival_test = np.array(
#         [(bool(e), float(t)) for e, t in zip(event_indicators, true_times)],
#         dtype=[('event', bool), ('time', float)]
#     )

#     # If training data provided, use it for censoring distribution estimation
#     if survival_train_times is not None and survival_train_events is not None:
#         survival_train_events = np.array(survival_train_events, dtype=bool)
#         survival_train_times = np.array(survival_train_times)
#         survival_train = np.array(
#             [(bool(e), float(t)) for e, t in zip(survival_train_events, survival_train_times)],
#             dtype=[('event', bool), ('time', float)]
#         )
#     else:
#         # Fall back to using test set (less ideal but functional)
#         survival_train = survival_test

#     # Compute integrated Brier score
#     ibs = integrated_brier_score(
#         survival_train,
#         survival_test,
#         survival_probs,
#         time_points
#     )
#     return ibs


def naive_brier_score(pmf, true_bins, event_indicators):
    """
    Compute Brier score for discrete/binned survival predictions.

    This computes the mean squared error between predicted and true event probabilities
    across all time bins, accounting for censoring.

    Args:
        pmf: (n, num_bins) predicted probability mass function
        true_bins: (n,) true event time bins (0-indexed)
        event_indicators: (n,) 1 if event observed, 0 if censored

    Returns:
        brier: float, mean Brier score across all time bins
    """
    pmf = np.array(pmf)
    true_bins = np.array(true_bins)
    event_indicators = np.array(event_indicators)

    n, num_bins = pmf.shape

    # Compute survival function from PMF: S(t) = P(T > t) = sum_{k>t} pmf(k)
    # Use reverse cumsum
    survival_pred = np.flip(np.cumsum(np.flip(pmf, axis=1), axis=1), axis=1)

    brier_scores = []

    for t in range(num_bins):
        # At each time bin t, compute Brier score for "event by time t"
        # True label: Y_t = 1 if (event occurred AND time <= t), 0 otherwise

        # For uncensored: Y_t = 1 if true_bins <= t, else 0
        # For censored: only include if censoring time > t (we know they survived past t)

        squared_errors = []

        for i in range(n):
            # Predicted probability of event by time t
            # P(T <= t) = 1 - S(t) = sum_{k<=t} pmf(k)
            p_event_by_t = 1.0 - survival_pred[i, t]

            if event_indicators[i] == 1:
                # Uncensored: we know the true event time
                y_true = 1.0 if true_bins[i] <= t else 0.0
                squared_errors.append((y_true - p_event_by_t) ** 2)
            else:
                # Censored at true_bins[i]
                # Only include if censoring time > t (we know they survived to at least t)
                if true_bins[i] > t:
                    # We know event didn't occur by time t
                    y_true = 0.0
                    squared_errors.append((y_true - p_event_by_t) ** 2)
                # If censoring time <= t, we can't use this sample for this time point

        if squared_errors:
            brier_scores.append(np.mean(squared_errors))

    # Integrated Brier score: average across all time bins
    return np.mean(brier_scores) if brier_scores else 0.0

def integrated_brier_score(pmf, true_times, event_indicators, bin_edges) -> float:
    """
    Compute integrated Brier score for discrete time survival predictions.
    Using sksurv.

    Args:
        pmf: (n, num_bins) predicted PMF 
        true_times: (n,) true event times
        event_indicators: (n,) 1 if event, 0 if censored
        bin_edges: (num_bins + 1,) edges of the time bins
        
        Returns:
        ibs: float, integrated Brier score
    """
    survival_train = np.array([(e, t) for e, t in zip(event_indicators, true_times)], dtype=[('event', bool), ('time', float)])
    survival_test = survival_train

    cif = pmf.cumsum(axis=1)
    survival_probs = 1.0 - cif
    
    min_time = survival_test["time"].min()
    max_time = survival_test["time"].max()

    times = bin_edges[1:]
    valid = (times > min_time) & (times < max_time)

    times = times[valid]
    survival_probs = survival_probs[:, valid]
    
    ibs = sk_integrated_brier_score(
        survival_train,
        survival_test,
        survival_probs,
        times
    )
    
    return ibs

def compute_calibration_error(pmf, true_bins, num_bins, num_prob_bins=10):
    """
    Compute mean absolute calibration error.

    Args:
        pmf: (n, num_bins) predicted PMF
        true_bins: (n,) true time bins
        num_bins: number of time bins
        num_prob_bins: number of probability bins for calibration

    Returns:
        calibration_error: float, mean absolute error
    """
    all_pred_probs = []
    all_outcomes = []

    for i in range(pmf.shape[0]):
        for k in range(num_bins):
            all_pred_probs.append(pmf[i, k])
            all_outcomes.append(1.0 if k == true_bins[i] else 0.0)

    all_pred_probs = np.array(all_pred_probs)
    all_outcomes = np.array(all_outcomes)

    # Bin by predicted probability
    prob_bins = np.linspace(0.0, 1.0, num_prob_bins + 1)
    bin_ids = np.digitize(all_pred_probs, prob_bins) - 1

    errors = []
    for b in range(num_prob_bins):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue

        mean_pred = all_pred_probs[idx].mean()
        empirical_freq = all_outcomes[idx].mean()
        errors.append(abs(mean_pred - empirical_freq))

    return np.mean(errors) if errors else 0.0

def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence D_KL(P || Q) for discrete distributions.

    Args:
        p: (batch_size, num_bins) true distribution
        q: (batch_size, num_bins) predicted distribution
    Returns:
        kl_div: (batch_size,) KL divergence for each sample
    """
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10
    kl_div = torch.sum(p * torch.log(p / q), dim=1)
    return torch.mean(kl_div, dim=0).cpu().numpy()

