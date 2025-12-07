import torch

def soft_label_distribution(event_times, num_bins, sigma_rel=0.05, device='cpu'):
    """
    Create soft target distributions for uncensored events using a Gaussian.
    
    Args:
        event_times: [batch_size] tensor of integer bin indices
        num_bins: total number of discrete bins
        sigma_rel: relative width of Gaussian smoothing (fraction of bins)
        device: torch device
    Returns:
        q: [batch_size, num_bins] soft target distributions
    """
    sigma = sigma_rel * num_bins
    t = torch.arange(num_bins, device=device).float()  # [num_bins]
    batch_size = event_times.size(0)
    
    q = torch.exp(-0.5 * ((t.unsqueeze(0) - event_times.unsqueeze(1).float()) / sigma) ** 2)
    q /= q.sum(dim=1, keepdim=True)  # normalize per example
    return q

def soft_NLL(outputs, cif, t, e, sigma_rel=0.05, eps=1e-8):
    """
    Compute distribution-aware DeepHit loss with soft labels.
    
    Args:
        outputs: [batch_size, num_bins] predicted probabilities (after softmax)
        t: [batch_size] observed time bins (integer indices)
        censored: [batch_size] boolean tensor, True if censored
        sigma_rel: relative width of Gaussian smoothing for uncensored events
        eps: small constant to avoid log(0)
    Returns:
        loss: scalar tensor
    """
    device = outputs.device
    num_bins = outputs.size(1)
    
    # --- Uncensored events ---
    uncensored_mask = e == 1
    if uncensored_mask.any():
        unc_outputs = outputs[uncensored_mask]  # [num_unc, num_bins]
        unc_t = t[uncensored_mask]              # [num_unc]
        
        # Soft labels for uncensored events
        soft_targets = soft_label_distribution(unc_t, num_bins, sigma_rel, device)
        
        # Cross-entropy with soft targets
        unc_loss = -(soft_targets * torch.log(unc_outputs.clamp(min=eps, max=1.0))).sum(dim=1).mean()
    else:
        unc_loss = torch.tensor(0.0, device=outputs.device)

    # --- Censored events ---
    censored_mask = e == 0
    if censored_mask.any():
        cens_cif = cif[censored_mask]
        cens_t = t[censored_mask]
        
        # Only want  minimise outputs before cenosring time in bins which are not the overflow bin
        last_bin_idx = outputs.shape[1] - 1
        cens_t[cens_t == last_bin_idx] = last_bin_idx - 1  # Prevent overflow bin issues
        
        cens_cif_t = cens_cif[torch.arange(cens_cif.size(0)), cens_t]
    
        survival = (1.0 - cens_cif_t).clamp(min=eps)
        
        cens_loss = -torch.log(survival)
        cens_loss = torch.mean(cens_loss) 
    else:
        cens_loss = torch.tensor(0.0, device=outputs.device)
    
    # Total loss
    loss = unc_loss + cens_loss
    return loss, cens_loss, unc_loss

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
        censored_loss: scalar tensor
        uncensored_loss: scalar_tensor
    """
    batch_size, num_bins = outputs.shape
    eps = 1e-6

    # --- Uncensored samples (events occurred) ---
    uncensored_mask = e == 1
    if uncensored_mask.any():
        unc_outputs = outputs[uncensored_mask]
        unc_t = t[uncensored_mask]

        # Event probability at event time
        o_t = unc_outputs[torch.arange(unc_outputs.size(0)), unc_t]

        # Cumulative probability up to each time bin
        # F_t = torch.cumsum(unc_outputs, dim=1)
        # F_t = torch.clamp(F_t, max=1.0 - eps)

        # s_t = torch.ones_like(o_t, device=o_t.device)
        # mask = unc_t > 0
        # s_t[mask] = 1.0 - F_t[torch.arange(F_t.size(0), device=o_t.device)[mask], unc_t[mask]-1]

        # ratio = (o_t / torch.clamp(s_t, min=eps)).clamp(min=eps, max=1.0)
        # uncensored_loss = -torch.log(ratio)
        # uncensored_loss = torch.mean(uncensored_loss)
        
        # True time bin only considrered
        uncensored_loss = torch.mean(-torch.log(o_t.clamp(min=0, max=1.0)))
    else:
        uncensored_loss = torch.tensor(0.0, device=outputs.device)

    # --- Censored samples (no event) ---
    censored_mask = e == 0
    if censored_mask.any():
        cens_cif = cif[censored_mask]
        cens_t = t[censored_mask]

        # CIF value at censoring time
        # Only want  minimise outputs before cenosring time in bins which are not the overflow bin
        last_bin_idx = outputs.shape[1] - 1
        cens_t[cens_t == last_bin_idx] = last_bin_idx - 1  # Prevent overflow bin issues
        
        cens_cif_t = cens_cif[torch.arange(cens_cif.size(0)), cens_t]
        
        # Clamp to [0, 1)
        # cens_cif_t = torch.clamp(cens_cif_t, min=0.0, max=1.0 - eps)
        survival = (1.0 - cens_cif_t).clamp(min=eps)
        
        #Dont care about minimising surviavl in final tiime_bin ("overflow bin")
        # last_bin_idx = outputs.shape[1] - 1
        # survival[cens_t==last_bin_idx] = 1
        
        censored_loss = -torch.log(survival)
        censored_loss = torch.mean(censored_loss) 
         # Mask final overflow bin
        # last_bin_idx = outputs.shape[1] - 1
        # mask = cens_t != last_bin_idx
        # censored_loss = torch.zeros_like(survival)
        # censored_loss[mask] = -torch.log(survival[mask])
        # censored_loss = torch.mean(censored_loss)
    else:
        censored_loss = torch.tensor(0.0, device=outputs.device)

    # total_loss = (uncensored_loss + censored_loss) / batch_size
    # uncensored_loss = uncensored_loss / max(uncensored_mask.sum(), 1)
    # censored_loss = censored_loss / max(censored_mask.sum(), 1)
    total_loss = uncensored_loss + censored_loss
    return total_loss, censored_loss, uncensored_loss

def ranking_loss(
    cif: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    sigma: float = 0.2
) -> torch.Tensor:
    """
    Ranking loss to encourage correct ordering of predicted risk scores.    
    Args:
        cif: (batch_size, num_time_bins) - cumulative incidence function (CIF) for each time bin
        t: (batch_size,) - true event/censoring times (as indices of time bins)
        t_last: (batch_size,) - last observed times (as indices of time bins)
        e: (batch_size,) - event indicators (1 if event occurred, 0 if censored)
        sigma: float - scaling parameter for the logistic function
    """

    batch_size, num_bins = cif.shape
    t_i = t.unsqueeze(1).expand(-1, batch_size) # (batch_size_i, batch_size_j)
    t_j = t.unsqueeze(0).expand(batch_size, -1) # (batch_size_i, batch_size_j)
    
    e_i = e.unsqueeze(1).expand(-1, batch_size) # (batch_size_i, batch_size_j)
    
    # F_ij = cif # (batch_size, num_bins)
    F_ii = cif[torch.arange(batch_size), t] # (batch_size,)
    F_ii = F_ii.unsqueeze(1).expand(batch_size, batch_size) # (batch_size_i, batch_size_j) expanded alonng dim1
    
    F_ij = cif[:, t].T # (batch_size_i, batch_size_j)
    def eta(a, b):
        return torch.exp(-(a-b)/sigma)
    
    cif_comparison = eta(F_ii, F_ij)
    A_ij = ((t_i < t_j) & (e_i==1))
    
    loss = torch.sum(A_ij * cif_comparison) / (torch.sum(A_ij) + 1e-8)
    
    return loss
    
    
def prediction_loss(y: torch.Tensor, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
    """MSE loss for RNN networks predictions at next time step.
    Args:
        y: (batch_size, seq_length, num_features) - true features 
        x: (batch_size, seq_length, num_features) - predicted features 
    """
    if mask is None:
        mask = torch.ones(x.shape)
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


