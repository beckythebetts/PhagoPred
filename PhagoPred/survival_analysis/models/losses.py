import torch

def negative_log_likelihood(outputs: torch.Tensor, 
                            cif: torch.Tensor, 
                            t: torch.Tensor, 
                            e: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of joint distribution of first hitting time and survival time.
    
    \begin{align*} {\mathcal L}_{1} = 
    - & \sum _{i=1}^{N} \left[{\mathbb {1}} (k^{i} \ne {\varnothing }) \cdot \log \left(\frac{o_{k^{i},\tau ^{i}}^{i}}{1-\sum _{k\ne {\varnothing }}\sum _{n\leq t^{i}_{J^{i}}} o_{k,n}^{i}}\right) \right. \nonumber\\ &+ \left. {\mathbb {1}}(k^{i}={\varnothing })\cdot \log \left(1- \sum _{k\ne {\varnothing }}\hat{F}_{k}(\tau ^{i}|{\mathcal X}^{i}) \right) \right], \tag{7} \end{align*}
    
    Args:
        outputs: (batch_size, num_time_bins) - predicted probability distribution over time bins
        cif: (batch_size, num_time_bins) - cumulative incidence function (CIF) for each time bin
        t: (batch_size,) - true event/censoring times (as indices of time bins)
        e: (batch_size,) - event indicators (1 if event occurred, 0 if censored)
    Returns:
        loss: scalar tensor - negative log-likelihood loss
    """
    batch_size, num_bins = outputs.shape
    eps = 1e-8

    # --- Uncensored samples (events occurred) ---
    uncensored_mask = e == 1
    if uncensored_mask.any():
        unc_outputs = outputs[uncensored_mask]
        unc_t = t[uncensored_mask]

        # Event probability at event time
        o_t = unc_outputs[torch.arange(unc_outputs.size(0)), unc_t]

        # Cumulative probability up to each time bin
        F_t = torch.cumsum(unc_outputs, dim=1)
        F_t = torch.clamp(F_t, max=1.0 - eps)

        # Survival probability at time t (must be >= event prob)
        s_t = 1.0 - F_t[torch.arange(F_t.size(0)), unc_t]

        # Ensure ordering: o_t <= s_t
        # If violated slightly due to rounding, cap ratio to 1
        ratio = (o_t / torch.clamp(s_t, min=eps)).clamp(min=eps, max=1.0)

        uncensored_loss = -torch.log(ratio)
        uncensored_loss = torch.sum(uncensored_loss)
    else:
        uncensored_loss = torch.tensor(0.0, device=outputs.device)

    # --- Censored samples (no event) ---
    censored_mask = e == 0
    if censored_mask.any():
        cens_cif = cif[censored_mask]
        cens_t = t[censored_mask]

        # CIF value at censoring time
        cens_cif_t = cens_cif[torch.arange(cens_cif.size(0)), cens_t]

        # Clamp to [0, 1)
        cens_cif_t = torch.clamp(cens_cif_t, min=0.0, max=1.0 - eps)

        censored_loss = -torch.log(1.0 - cens_cif_t)
        censored_loss = torch.sum(censored_loss)
    else:
        censored_loss = torch.tensor(0.0, device=outputs.device)

    # --- Combine and normalize ---
    total_loss = (uncensored_loss + censored_loss) / batch_size
    total_loss = torch.clamp(total_loss, min=0.0)  # never negative
    return total_loss

def ranking_loss(
    cif: torch.Tensor,
    t: torch.Tensor,
    t_last: torch.Tensor,
    e: torch.Tensor,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    Ranking loss to encourage correct ordering of predicted risk scores.
    \begin{equation*} {\mathcal L}_{2} = \sum _{k=1}^{K} \alpha _{k} \sum _{i \ne j} A_{kij} \cdot \eta \left(\hat{F}_{k}(s^{i}+t^{i}_{J^{i}}|{\mathcal X}^{i}), \hat{F}_{k}(s^{i}+t^{j}_{J^{j}}|{\mathcal X}^{j})\right), \tag{8} \end{equation*}
    
    Args:
        cif: (batch_size, num_time_bins) - cumulative incidence function (CIF) for each time bin
        t: (batch_size,) - true event/censoring times (as indices of time bins)
        t_last: (batch_size,) - last observed times (as indices of time bins)
        e: (batch_size,) - event indicators (1 if event occurred, 0 if censored)
        sigma: float - scaling parameter for the logistic function
    """
    s = t - t_last  # (batch_size,)
    batch_size = cif.size(0)
    num_time_bins = cif.size(1)
    
    F_i = cif[torch.arange(batch_size), t]  # (batch_size_i,)
    F_i = F_i.unsqueeze(1).expand(-1, batch_size)  # (batch_size_i, batch_size_j)
    
    t_last_j = t_last.unsqueeze(0).expand(batch_size, -1)  # (batch_size_i, batch_size_j)
    s_i = s.unsqueeze(1).expand(-1, batch_size)  # (batch_size_i, batch_size_j)
    s_j = s.unsqueeze(0).expand(batch_size, -1)  # (batch_size_i, batch_size_j)
    e_i = e.unsqueeze(1).expand(-1, batch_size)  # (batch_size_i, batch_size_j)
    
    idxs_j = torch.arange(batch_size).unsqueeze(0).expand(batch_size, -1)  # (batch_size_i, batch_size_j)
    F_j = cif[idxs_j, (s_i + t_last_j).clamp(0, num_time_bins - 1)]  # (batch_size_i, batch_size_j)
    
    def eta(a, b):
        return torch.exp(-(a-b)/sigma)
    
    cif_comparisons = eta(F_i, F_j)  # (batch_size_i, batch_size_j)

    A_ij = ((s_i < s_j) & (e_i == 1)).float()  # Valid mask (batch_size_i, batch_size_j)
    # A_ij = (e_i ==1).float()

    loss = torch.sum(A_ij * cif_comparisons) / (torch.sum(A_ij) + 1e-8)
    
    # valid_pairs = torch.sum(A_ij)
    # print(valid_pairs)
    return loss

def prediction_loss(y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE loss for RNN networks predictions at next time step.
    Args:
        y: (batch_size, seq_length, num_features) - true features 
        x: (batch_size, seq_length, num_features) - predicted features 
    """
    x_shift = x[:, :-1, :]  # (batch_size, seq_length-1, num_features)
    y_shift = y[:, 1:, :]   # (batch_size, seq_length-1, num_features)
    mask_shift = mask[:, 1:, :]      # (batch_size, seq_length-1, num_features)
    
    mse_loss = torch.nn.functional.mse_loss(x_shift, y_shift, reduction='none')
    masked_loss = mse_loss * mask_shift
    return masked_loss.sum() / (mask_shift.sum() + 1e-8)


def test(num_samples=10, num_time_bins=20):
    #test_inputs:
    num_samples = 10
    num_time_bins = 20

    # Random logits converted to probabilities for outputs
    logits = torch.randn(num_samples, num_time_bins)
    test_outputs = torch.softmax(logits, dim=1)  # Valid probability distributions over time bins

    # CIF should be cumulative sum of some non-negative values <= 1
    # Create random non-negative increments and cumsum them to simulate CIF
    increments = torch.rand(num_samples, num_time_bins)
    increments = increments / increments.sum(dim=1, keepdim=True)  # Normalize increments so sum = 1
    test_cif = torch.cumsum(increments, dim=1)
    test_cif = torch.clamp(test_cif, max=1.0)  # Clamp to 1 just in case

    # Event/censoring times as indices within time bins
    test_t = torch.randint(0, num_time_bins, (num_samples,))
    test_t_last = torch.randint(0, num_time_bins, (num_samples,))

    # Binary event indicator (0 = censored, 1 = event)
    test_e = torch.randint(0, 2, (num_samples,))
    print(negative_log_likelihood(test_outputs, test_cif, test_t, test_e))
    print(ranking_loss(test_cif, test_t, test_t_last, test_e))
    
    
def main():
    test()
    
    
if __name__ == "__main__":
    main()


