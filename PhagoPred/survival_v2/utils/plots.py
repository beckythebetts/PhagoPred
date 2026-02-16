import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def visualise_prediction(sample, predicted_pmf, bin_edges, attn_weights=None, feature_names=None, save_path=None, start_frame=0) -> None:
    """
    Plot observed features, attention weights (if available) and predicted (and underlying) PMF.

    Args:
        sample: dict with keys:
            - features: (seq_len, num_features) tensor/array of observed features
            - time_to_event_bin: scalar tensor/value of true time bin
            - event_indicator: scalar tensor/value, 1 if event, 0 if censored
            - (optional) binned_pmf: true PMF distribution
        predicted_pmf: (num_bins,) tensor/array of predicted PMF
        bin_edges: bin edge positions
        attn_weights: (seq_len,) tensor/array of attention weights, or None
        feature_names: list of feature names, or None
        save_path: Path to save figure, or None to not save
    """

    # Helper function to convert to numpy if needed
    def to_numpy(x):
        if x is None:
            return None
        if hasattr(x, 'cpu'):  # torch tensor
            return x.cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    # Convert all inputs to numpy arrays
    features = to_numpy(sample['features'])
    true_pmf = to_numpy(sample.get('binned_pmf'))
    time_to_event = to_numpy(sample['time_to_event'])
    cell_idx = sample.get('cell_idx', 'N/A')
    cell_file = sample.get('hdf5_path', 'N/A')
    length = to_numpy(sample.get('length', None))
    landmark_frame = to_numpy(sample.get('landmark_frame', 0))
    if length is None or (isinstance(length, np.ndarray) and length.size == 0):
        length = len(features)
    elif isinstance(length, np.ndarray):
        length = int(length.item())
    else:
        length = int(length)
    event_indicator = to_numpy(sample['event_indicator'])
    if isinstance(event_indicator, np.ndarray):
        event_indicator = event_indicator.item()

    attn_weights = to_numpy(attn_weights)
    predicted_pmf = to_numpy(predicted_pmf)

    padding_slice = slice(None, length)

    # Adjust layout based on whether we have attention weights
    if attn_weights is not None:
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1.2])

        # ====== (1) Attention Values ======
        attn_weights_trimmed = attn_weights[padding_slice]
        ax_att = fig.add_subplot(gs[0, 0])
        ax_att.scatter(np.arange(length), attn_weights_trimmed, color='k', marker='o', label='Attention Weights')
        ax_att.set_ylabel("Attention")

        # ===== (2) Features =====
        ax_feat = fig.add_subplot(gs[1, 0])
    else:
        # No attention weights - simpler layout
        fig = plt.figure(figsize=(14, 4))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_feat = fig.add_subplot(gs[0, 0])

    # ===== Features =====
    features = features[:length, :]
    for f_idx in range(features.shape[1]):
        ax_feat.plot(np.arange(start_frame, start_frame+length), features[:, f_idx], label=(feature_names[f_idx] if feature_names else f"F{f_idx}"), alpha=0.7)

    ax_feat.set_xlabel("Time / frames")
    ax_feat.set_ylabel("Features")
    ax_feat.legend(fontsize=8, ncol=2)
    ax_feat.grid(True)

    # ====== PMFS ======
    if attn_weights is not None:
        ax_sd = fig.add_subplot(gs[:, 1])  # span both rows
    else:
        ax_sd = fig.add_subplot(gs[0, 1])
    abs_bin_edges = bin_edges + landmark_frame
    bin_widths = np.diff(abs_bin_edges)
    bin_widths[-1] = bin_widths[-2]  # make last bin same width for visualization

    ax_sd.bar(
        abs_bin_edges[:-1],
        predicted_pmf,
        width=bin_widths,
        align='edge',
        color='tab:blue',
        edgecolor='k',
        alpha=0.5,
        label='Predicted PMF',
    )

    if true_pmf is not None:
        ax_sd.bar(
            abs_bin_edges[:-1],
            true_pmf,
            width=bin_widths,
            align='edge',
            color='tab:red',
            edgecolor='k',
            alpha=0.5,
            label='True PMF',
            )

    abs_event_time = time_to_event + landmark_frame
    if event_indicator == 1:
        ax_sd.axvline(abs_event_time, color='red', linestyle='--', label="Event Time")
    else:
        ax_sd.axvline(abs_event_time, color='orange', linestyle='--', label="Censoring Time")

    # ax_sd.set_title(f"Predicted Survival Distribution {cell_idx}, {cell_file}")
    ax_sd.set_xlabel("Time / frames")
    ax_sd.set_ylabel("Probability")
    ax_sd.legend()
    ax_sd.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
def plot_cm(cm, save_path) -> None:
    cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalised, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.xlabel("Predicted Time Bin")
    plt.ylabel("True Time Bin")
    plt.title("Confusion Matrix of Predicted vs True Time-to-Event Bins")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()


def plot_soft_confusion_matrix(pred_pmfs, true_bins, save_path=None) -> None:
    """
    Soft confusion matrix showing average predicted probability mass for each true bin.
    Instead of argmax predictions, shows the full distribution.

    Args:
        pred_pmfs: (n_samples, num_bins) - predicted PMFs
        true_bins: (n_samples,) - true time bins
        save_path: Path to save figure

    This answers: "When the true bin is X, how much probability mass does
    the model typically assign to each bin?"
    """
    num_bins = pred_pmfs.shape[1]

    # Compute average predicted PMF for each true bin
    soft_cm = np.zeros((num_bins, num_bins))
    counts = np.zeros(num_bins)

    for true_bin in range(num_bins):
        mask = true_bins == true_bin
        if mask.sum() > 0:
            soft_cm[true_bin, :] = pred_pmfs[mask].mean(axis=0)
            counts[true_bin] = mask.sum()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a diverging colormap centered at 0
    im = ax.imshow(soft_cm, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Predicted Probability', rotation=270, labelpad=20)

    # Labels
    ax.set_xlabel('Predicted Time Bin', fontsize=12)
    ax.set_ylabel('True Time Bin', fontsize=12)
    ax.set_title('Soft Confusion Matrix\n(Average Predicted PMF by True Bin)', fontsize=14)

    # Set ticks
    ax.set_xticks(np.arange(num_bins))
    ax.set_yticks(np.arange(num_bins))
    ax.set_xticklabels(np.arange(num_bins))
    ax.set_yticklabels([f'{i} (n={int(counts[i])})' for i in range(num_bins)])

    # Add text annotations
    for i in range(num_bins):
        for j in range(num_bins):
            if counts[i] > 0:
                text = ax.text(j, i, f'{soft_cm[i, j]:.2f}',
                             ha="center", va="center", color="black" if soft_cm[i, j] < 0.5 else "white",
                             fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_calibration_curve(pred_pmfs, true_bins, num_bins, save_path=None) -> None:
    """
    Calibration curve showing predicted vs empirical probabilities.
    Checks if predicted probabilities match actual frequencies.

    Args:
        pred_pmfs: (n_samples, num_bins) - predicted PMFs
        true_bins: (n_samples,) - true time bins
        num_bins: number of time bins
        save_path: Path to save figure

    This answers: "When the model predicts probability P for a bin,
    does that bin actually occur P fraction of the time?"
    """
    # Collect all (predicted_prob, actual_outcome) pairs
    pred_probs = []
    outcomes = []

    for i in range(len(pred_pmfs)):
        for bin_idx in range(num_bins):
            pred_probs.append(pred_pmfs[i, bin_idx])
            outcomes.append(1.0 if true_bins[i] == bin_idx else 0.0)

    pred_probs = np.array(pred_probs)
    outcomes = np.array(outcomes)

    # Bin predictions and compute empirical frequencies
    num_prob_bins = 10
    prob_bins = np.linspace(0, 1, num_prob_bins + 1)

    bin_centers = []
    empirical_freqs = []
    bin_counts = []

    for i in range(num_prob_bins):
        mask = (pred_probs >= prob_bins[i]) & (pred_probs < prob_bins[i+1])
        if i == num_prob_bins - 1:  # Include right edge
            mask |= (pred_probs == 1.0)

        if mask.sum() > 0:
            bin_centers.append((prob_bins[i] + prob_bins[i+1]) / 2)
            empirical_freqs.append(outcomes[mask].mean())
            bin_counts.append(mask.sum())

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Actual calibration
    ax.scatter(bin_centers, empirical_freqs, s=[c/10 for c in bin_counts],
              alpha=0.6, color='tab:blue', edgecolors='black', linewidths=1,
              label='Model calibration')

    # Connect with line
    if len(bin_centers) > 1:
        ax.plot(bin_centers, empirical_freqs, 'tab:blue', alpha=0.5, linewidth=1)

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Empirical Frequency', fontsize=12)
    ax.set_title('Calibration Curve\n(bubble size = number of predictions)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_pmf_comparison_grid(pred_pmfs, true_pmfs, true_bins, bin_edges, num_examples=16, save_path=None) -> None:
    """
    Grid of example predictions showing predicted vs true PMFs.

    Args:
        pred_pmfs: (n_samples, num_bins) - predicted PMFs
        true_pmfs: (n_samples, num_bins) - true PMFs (can be None)
        true_bins: (n_samples,) - true time bins (for vertical line)
        bin_edges: bin edge positions
        num_examples: number of examples to show
        save_path: Path to save figure

    This answers: "What do individual predictions look like compared to truth?"
    """
    num_examples = min(num_examples, len(pred_pmfs))

    # Select diverse examples (spread across true bins)
    unique_bins = np.unique(true_bins)
    examples_per_bin = max(1, num_examples // len(unique_bins))

    selected_indices = []
    for bin_idx in unique_bins:
        bin_mask = true_bins == bin_idx
        bin_indices = np.where(bin_mask)[0]
        if len(bin_indices) > 0:
            selected = np.random.choice(bin_indices,
                                       size=min(examples_per_bin, len(bin_indices)),
                                       replace=False)
            selected_indices.extend(selected)

    selected_indices = selected_indices[:num_examples]

    # Create grid
    ncols = 4
    nrows = (num_examples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
    axes = axes.flatten() if num_examples > 1 else [axes]

    num_bins = pred_pmfs.shape[1]
    bin_widths = np.diff(bin_edges)

    for plot_idx, sample_idx in enumerate(selected_indices):
        ax = axes[plot_idx]

        # Plot bars
        ax.bar(bin_edges[:-1], pred_pmfs[sample_idx], width=bin_widths,
              align='edge', alpha=0.6, color='tab:blue', edgecolor='black',
              label='Predicted PMF')

        if true_pmfs is not None:
            ax.bar(bin_edges[:-1], true_pmfs[sample_idx], width=bin_widths,
                  align='edge', alpha=0.4, color='tab:red', edgecolor='black',
                  label='True PMF')

        # True event time
        true_time = bin_edges[true_bins[sample_idx]]
        ax.axvline(true_time, color='red', linestyle='--', linewidth=2, label='True event')

        ax.set_xlabel('Time')
        ax.set_ylabel('Probability')
        ax.set_title(f'Sample {sample_idx} (true bin {true_bins[sample_idx]})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(pred_pmfs[sample_idx].max(),
                          true_pmfs[sample_idx].max() if true_pmfs is not None else 0) * 1.1)

    # Hide unused subplots
    for plot_idx in range(len(selected_indices), len(axes)):
        axes[plot_idx].axis('off')

    plt.suptitle('Predicted vs True PMF Distributions', fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def plot_spread_analysis(pred_pmfs, true_pmfs, true_bins, save_path=None) -> None:
    """
    Analyze and visualize the spread (entropy/variance) of predictions.

    Args:
        pred_pmfs: (n_samples, num_bins) - predicted PMFs
        true_pmfs: (n_samples, num_bins) - true PMFs (can be None)
        true_bins: (n_samples,) - true time bins
        save_path: Path to save figure

    This answers: "Is the model producing appropriately spread-out distributions?"
    """
    def compute_entropy(pmfs):
        """Compute Shannon entropy for each PMF"""
        eps = 1e-8
        return -np.sum(pmfs * np.log(pmfs + eps), axis=1)

    def compute_std(pmfs):
        """Compute standard deviation for each PMF"""
        num_bins = pmfs.shape[1]
        bins = np.arange(num_bins)
        mean = (pmfs * bins).sum(axis=1)
        variance = (pmfs * (bins - mean[:, np.newaxis])**2).sum(axis=1)
        return np.sqrt(variance)

    pred_entropy = compute_entropy(pred_pmfs)
    pred_std = compute_std(pred_pmfs)

    if true_pmfs is not None:
        true_entropy = compute_entropy(true_pmfs)
        true_std = compute_std(true_pmfs)

    # Create figure
    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # 1. Entropy distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pred_entropy, bins=30, alpha=0.6, color='tab:blue', label='Predicted', density=True)
    if true_pmfs is not None:
        ax1.hist(true_entropy, bins=30, alpha=0.6, color='tab:red', label='True', density=True)
    ax1.set_xlabel('Entropy (nats)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('PMF Entropy Distribution', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Standard deviation distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(pred_std, bins=30, alpha=0.6, color='tab:blue', label='Predicted', density=True)
    if true_pmfs is not None:
        ax2.hist(true_std, bins=30, alpha=0.6, color='tab:red', label='True', density=True)
    ax2.set_xlabel('Standard Deviation (bins)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('PMF Spread Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Spread vs true bin (check if spread varies by time)
    ax3 = fig.add_subplot(gs[0, 2])
    num_bins = pred_pmfs.shape[1]
    bin_means = []
    bin_stds_pred = []
    bin_stds_true = []

    for bin_idx in range(num_bins):
        mask = true_bins == bin_idx
        if mask.sum() > 0:
            bin_means.append(bin_idx)
            bin_stds_pred.append(pred_std[mask].mean())
            if true_pmfs is not None:
                bin_stds_true.append(true_std[mask].mean())

    ax3.plot(bin_means, bin_stds_pred, 'o-', color='tab:blue', label='Predicted', linewidth=2, markersize=8)
    if true_pmfs is not None:
        ax3.plot(bin_means, bin_stds_true, 'o-', color='tab:red', label='True', linewidth=2, markersize=8)
    ax3.set_xlabel('True Time Bin', fontsize=11)
    ax3.set_ylabel('Average Std Dev (bins)', fontsize=11)
    ax3.set_title('Spread vs True Bin', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle('PMF Spread Analysis', fontsize=16, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
def plot_losses(loss_history, save_path) -> None:
    """
    Plot training and validation losses
    Args:
        loss_history: list of dicts with keys 'train' and 'val', each a list of loss values per epoch
        save_path: Path to save figure, or None to not save
    """
    epochs = np.arange(1, len(loss_history) + 1)

    loss_types = list(loss_history[0]['train'].keys())

    train_losses = {lt: [] for lt in loss_types}
    val_losses = {lt: [] for lt in loss_types}

    for epoch_data in loss_history:
        for lt in loss_types:
            train_losses[lt].append(epoch_data['train'][lt])
            val_losses[lt].append(epoch_data['val'][lt])

    # Filter out loss types that are always zero (not used in the model)
    active_loss_types = []
    for lt in loss_types:
        if lt == 'total':
            continue
        # Check if loss is non-zero in any epoch
        if any(train_losses[lt][i] > 0 or val_losses[lt][i] > 0 for i in range(len(epochs))):
            active_loss_types.append(lt)

    plt.figure(figsize=(10, 6))

    def plot_loss(ax, lt, colour):
        ax.plot(epochs, train_losses[lt], label=f'{lt.capitalize()} train loss', color=colour, linestyle='-')
        ax.plot(epochs, val_losses[lt], label=f'{lt.capitalize()} validation', color=colour, linestyle='--')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{lt} Loss")
        ax.set_title(f"{lt} Loss over Epochs")
        ax.legend()
        ax.grid(True)

    cmap = plt.get_cmap('Set1')

    for i, lt in enumerate(active_loss_types):
        plot_loss(plt.gca(), lt, cmap(i))
    plot_loss(plt.gca(), 'total', 'black')

    plt.tight_layout()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid=True
    plt.title("Training and Validation Losses over Epochs")
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    
    
    