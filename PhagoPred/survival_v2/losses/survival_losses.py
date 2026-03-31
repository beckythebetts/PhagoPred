import torch

from PhagoPred.utils.logger import get_logger

log = get_logger()


def hazard_nll(logits, t, e, bin_weights=None):
    h = torch.sigmoid(logits)
    eps = 1e-8
    log_h = torch.log(h.clamp(min=eps))
    log_1mh = torch.log((1 - h).clamp(min=eps))
    log_s = log_1mh.cumsum(dim=1)  # log S(t)
    t = t.long()
    num_bins = logits.shape[1]

    # Uncensored: log PMF(t) = log h(t) + log S(t-1)
    unc_mask = e == 1
    if unc_mask.any():
        idx = torch.arange(unc_mask.sum(), device=logits.device)
        unc_t = t[unc_mask]
        log_ht = log_h[unc_mask][idx, unc_t]
        log_s_prev = torch.where(
            unc_t > 0, log_s[unc_mask][idx, (unc_t - 1).clamp(min=0)],
            torch.zeros(unc_mask.sum(), device=logits.device))
        nll = -(log_ht + log_s_prev)
        if bin_weights is not None:
            w = torch.tensor(bin_weights,
                             dtype=torch.float32,
                             device=logits.device)[unc_t]
            uncensored_loss = (nll * w).sum() / w.sum()
        else:
            uncensored_loss = nll.mean()
    else:
        uncensored_loss = torch.tensor(0.0, device=logits.device)

    # Censored: log S(t), capped at last bin
    cens_mask = e == 0
    if cens_mask.any():
        cens_t = t[cens_mask].clamp(max=num_bins - 1)
        idx = torch.arange(cens_mask.sum(), device=logits.device)
        log_st = log_s[cens_mask][idx, cens_t]
        censored_loss = -log_st.mean()
    else:
        censored_loss = torch.tensor(0.0, device=logits.device)

    return uncensored_loss + censored_loss, censored_loss, uncensored_loss


def negative_log_likelihood(
    pmf: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    bin_weights: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Negative log-likelihood for discrete-time survival with censoring.

    Args:
        pmf: (batch_size, num_time_bins) - probability mass function
        t: (batch_size,) - event/censoring time bins (integers)
        e: (batch_size,) - event indicators (1=event, 0=censored)

    Returns:
        tuple: (total_loss, censored_loss, uncensored_loss)
    """
    if bin_weights is None:
        bin_weights = torch.tensor([])
    # batch_size, num_bins = pmf.shape
    cif = torch.cumsum(pmf, dim=1)
    eps = 1e-6

    # Ensure t is long for indexing
    t = t.long()

    # Uncensored events
    uncensored_mask = e == 1
    # log.info(f'Uncensored Mask: {e.size()}')
    if uncensored_mask.any():
        unc_pmf = pmf[uncensored_mask]
        unc_t = t[uncensored_mask]
        o_t = unc_pmf[torch.arange(unc_pmf.size(0), device=pmf.device), unc_t]
        nll = -torch.log(o_t.clamp(min=eps, max=1.0))
        if len(bin_weights) == 0:
            uncensored_loss = torch.mean(nll)
        else:
            w = bin_weights[unc_t]
            uncensored_loss = (nll * w).sum() / w.sum()

        # uncensored_loss = torch.mean(uncens)
    else:
        uncensored_loss = torch.tensor(0.0, device=pmf.device)

    # Censored observations
    censored_mask = e == 0
    if censored_mask.any():
        cens_cif = cif[censored_mask]
        cens_t = t[censored_mask]

        # Survival probability at censoring time (CIF at t-1)
        cens_t = torch.clamp(cens_t - 1, min=0)
        cens_cif_t = cens_cif[
            torch.arange(cens_cif.size(0), device=pmf.device), cens_t]
        survival = 1.0 - cens_cif_t
        survival = torch.clamp(survival, min=eps, max=1.0)

        censored_loss = -torch.log(survival)
        censored_loss = torch.mean(censored_loss)
    else:
        censored_loss = torch.tensor(0.0, device=pmf.device)

    total_loss = uncensored_loss + censored_loss
    return total_loss, censored_loss, uncensored_loss


def ranking_loss_concordance(pmf: torch.Tensor,
                             t: torch.Tensor,
                             e: torch.Tensor,
                             sigma: float = 0.2) -> torch.Tensor:
    """
    Concordance-based ranking loss using median survival time.
    For pairs (i,j) where t_i < t_j and i is uncensored:
    Penalizes when predicted median_i >= median_j.

    Args:
        pmf: (batch_size, num_time_bins) - probability mass function
        t: (batch_size,) - event/censoring time bins
        e: (batch_size,) - event indicators (1=event, 0=censored)
        sigma: temperature parameter for soft ranking

    Returns:
        ranking_loss: scalar tensor
    """
    cif = torch.cumsum(pmf, dim=1)
    median_time = torch.argmax((cif >= 0.5).float(), dim=1).float()

    batch_size = t.shape[0]
    t_i = t.unsqueeze(1).expand(-1, batch_size)
    t_j = t.unsqueeze(0).expand(batch_size, -1)
    e_i = e.unsqueeze(1).expand(-1, batch_size)

    med_i = median_time.unsqueeze(1).expand(batch_size, batch_size)
    med_j = median_time.unsqueeze(0).expand(batch_size, batch_size)

    # Only consider comparable pairs
    comparable = ((t_i < t_j) & (e_i == 1))

    # Penalize when median_i >= median_j
    violation = torch.nn.functional.relu(med_i - med_j + 1.0)

    return (comparable * violation).sum() / (comparable.sum() + 1e-8)


def ranking_loss_cif(pmf: torch.Tensor,
                     t: torch.Tensor,
                     e: torch.Tensor,
                     sigma: float = 0.2) -> torch.Tensor:
    """
    Original DeepHit-style ranking loss using CIF comparison.
    WARNING: May push mass to final bins in single-cause problems.

    For pairs (i,j) where t_i < t_j and i is uncensored:
    Wants CIF_i(t_i) > CIF_j(t_i).

    Args:
        pmf: (batch_size, num_time_bins) - probability mass function
        t: (batch_size,) - event/censoring time bins
        e: (batch_size,) - event indicators
        sigma: temperature parameter

    Returns:
        ranking_loss: scalar tensor
    """
    cif = torch.cumsum(pmf, dim=1)
    batch_size, num_bins = cif.shape

    # Ensure t is long for indexing
    t = t.long()

    t_i = t.unsqueeze(1).expand(-1, batch_size)
    t_j = t.unsqueeze(0).expand(batch_size, -1)
    e_i = e.unsqueeze(1).expand(-1, batch_size)

    # CIF at event times
    F_ii = cif[torch.arange(batch_size, device=pmf.device), t]
    F_ii = F_ii.unsqueeze(1).expand(batch_size, batch_size)
    F_ij = cif[:, t].T

    # Exponential penalty
    cif_comparison = torch.exp(-(F_ii - F_ij) / sigma)
    A_ij = ((t_i < t_j) & (e_i == 1))

    return torch.sum(A_ij * cif_comparison) / (torch.sum(A_ij) + 1e-8)


def soft_target_nll(
        pmf: torch.Tensor,
        t: torch.Tensor,
        e: torch.Tensor,
        sigma: float = 0.8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Soft target negative log-likelihood that penalizes by distance from true bin.
    Uses Gaussian-weighted targets centered at the true bin to encourage spread
    while penalizing distant predictions more than nearby ones.

    This is better than standard NLL for stochastic survival data where the true
    time is inherently uncertain. It works without needing true PMFs, making it
    suitable for real data.

    Args:
        pmf: (batch_size, num_time_bins) - predicted probability mass function
        t: (batch_size,) - event/censoring time bins (integers)
        e: (batch_size,) - event indicators (1=event, 0=censored)
        sigma: spread parameter for soft targets (default 0.8)
               - 0.3: very tight (99% on true bin)
               - 0.5: tight (79% on true, 21% on adjacent)
               - 0.8: medium (50% on true, 46% on adjacent) <- RECOMMENDED
               - 1.0: wide (40% on true, 49% on adjacent)

    Returns:
        tuple: (total_loss, censored_loss, uncensored_loss)

    Example:
        For true_bin=2, sigma=0.8, and 5 bins:
        Soft targets: [0.135, 0.325, 0.500, 0.325, 0.135]

        Comparison with standard NLL for different predictions:
        1. Perfect peaked [0, 0, 1, 0, 0]:
           Standard NLL: 0.041, Soft NLL: 2.327 (penalized for being too narrow)
        2. Spread around true [0.1, 0.3, 0.4, 0.15, 0.05]:
           Standard NLL: 0.916, Soft NLL: 1.222 (rewarded for matching spread)
        3. One bin off [0, 0, 0, 1, 0]:
           Standard NLL: 1.609, Soft NLL: 1.718 (mild penalty)
        4. Four bins off [1, 0, 0, 0, 0]:
           Standard NLL: 3.507, Soft NLL: 3.342 (heavy penalty)
    """
    batch_size, num_bins = pmf.shape
    t = t.long()
    eps = 1e-8

    # Create soft targets: Gaussian centered at true bin
    bins = torch.arange(num_bins, device=pmf.device).float()
    bins = bins.unsqueeze(0).expand(batch_size, -1)  # (batch, num_bins)
    t_expanded = t.unsqueeze(1).float()  # (batch, 1)

    # Gaussian weights
    soft_targets = torch.exp(-0.5 * ((bins - t_expanded) / sigma)**2)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)

    # Uncensored events: cross-entropy with soft targets
    uncensored_mask = e == 1
    if uncensored_mask.any():
        unc_pmf = pmf[uncensored_mask]
        unc_targets = soft_targets[uncensored_mask]

        # Cross-entropy: -sum(target * log(pred))
        uncensored_loss = -(unc_targets *
                            torch.log(unc_pmf.clamp(min=eps))).sum(dim=1)
        uncensored_loss = uncensored_loss.mean()
    else:
        uncensored_loss = torch.tensor(0.0, device=pmf.device)

    # Censored events: same as standard NLL (survival probability)
    censored_mask = e == 0
    if censored_mask.any():
        cif = torch.cumsum(pmf, dim=1)
        cens_cif = cif[censored_mask]
        cens_t = t[censored_mask]

        # Survival probability at censoring time (CIF at t-1)
        cens_t = torch.clamp(cens_t - 1, min=0)
        cens_cif_t = cens_cif[
            torch.arange(cens_cif.size(0), device=pmf.device), cens_t]
        survival = 1.0 - cens_cif_t
        survival = torch.clamp(survival, min=eps, max=1.0)

        censored_loss = -torch.log(survival).mean()
    else:
        censored_loss = torch.tensor(0.0, device=pmf.device)

    total_loss = uncensored_loss + censored_loss
    return total_loss, censored_loss, uncensored_loss


def prediction_loss(y_pred: torch.Tensor,
                    y_true: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
    """
    MSE loss for next-timestep feature prediction (LSTM predictor).

    Args:
        y_pred: (batch_size, seq_len, num_features) - predicted features
        y_true: (batch_size, seq_len, num_features) - true features
        mask: (batch_size, seq_len, num_features) - validity mask

    Returns:
        mse_loss: scalar tensor
    """
    if mask is None:
        mask = torch.ones_like(y_true)

    # Shift for next-timestep prediction
    y_pred_shift = y_pred[:, :-1, :]
    y_true_shift = y_true[:, 1:, :]
    mask_shift = mask[:, 1:, :]

    mse = torch.nn.functional.mse_loss(y_pred_shift,
                                       y_true_shift,
                                       reduction='none')
    masked_loss = mse * mask_shift

    return masked_loss.sum() / (mask_shift.sum() + 1e-8)


# # Loss combination presets
# LOSS_CONFIGS = {
#     'nll_only': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.8,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target_tight': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.5,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target_wide': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 1.0,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'nll_ranking_concordance': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.1,
#         'ranking_type': 'concordance',
#         'prediction': 0.0
#     },
#     'soft_target_ranking': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.8,
#         'ranking': 0.05,
#         'ranking_type': 'concordance',
#         'prediction': 0.0
#     },
#     'nll_ranking_cif': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.1,
#         'ranking_type': 'cif',
#         'prediction': 0.0
#     },
#     'nll_prediction': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.0,
#         'prediction': 0.5
#     },
#     'full': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.1,
#         'ranking_type': 'concordance',
#         'prediction': 0.5
#     }
# }
