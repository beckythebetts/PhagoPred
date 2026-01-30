from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import json
import numpy as np

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models.temp_cnn import model

def validate_step(model_, dataloader, loss_fn, device):
    model_.eval()
    # total_loss = 0.0
    losses = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:

            outputs = model_(batch['features'], batch['length'])
            loss_values = loss_fn(outputs, batch['time_to_event_bin'], batch['event_indicator'])
            for key, value in zip(
                ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Censored Loss', 'Uncensored Loss'], loss_values):
                losses[key] += value.item() * batch['features'].size(0)  # Multiply by batch size

    avg_losses = {key: value / len(dataloader.dataset) for key, value in losses.items()}
    return avg_losses

def saliency_map(model, x, length, n_samples=20, sigma_ratio=0.1):
    """
    SmoothGrad saliency for a single sequence (B=1)
    x: (T, C)
    length: scalar
    """
    model.eval()
    saliency_accum = torch.zeros_like(x)

    # Forward to get target bin
    with torch.no_grad():
        outputs = model(x, length)  # (1, num_bins)
    target_bin = int(outputs.argmax(dim=1).item())

    for _ in range(n_samples):
        sigma = sigma_ratio * (outputs.max() - outputs.min())
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).detach().clone().requires_grad_(True)

        logits = model(x_noisy, length, return_logits=True)
        scalar = logits[0, target_bin]

        model.zero_grad()
        scalar.backward()
        saliency_accum += x_noisy.grad.detach()

    saliency_map = saliency_accum / n_samples

    return saliency_map[0].cpu().numpy()  # (T, C)


def visualize_validation_predictions(
    model_,
    dataloader,
    device,
    bin_edges,
    features=None,
    num_examples=20,
    save_path=None,
    padding_at: str = 'start',
):
    import matplotlib.pyplot as plt

    model_.eval()
    model_.to(device)
    examples_plotted = 0

    for batch in dataloader:

        lengths = batch['length'].cpu().numpy()
        pmfs = batch['binned_pmf']
        time_to_events = batch['time_to_event'].cpu().numpy()
        event_indicators = batch['event_indicator'].cpu().numpy()
        cell_idxs = batch['cell_idx']
        files = batch['hdf5_path']
        feature_values = batch['features'].cpu().numpy()


        with torch.no_grad():
            outputs, attn_weights = model_(batch['features'], batch['length'], return_attention=True)

        predicted_pmf = model.estimated_pmf(outputs)
        attn_weights_np = attn_weights.cpu().numpy()

        for i in range(len(batch['features'])):

            seq_len = lengths[i]

            if padding_at == 'start':
                padding_slice = slice(-seq_len, None)
            else:
                padding_slice = slice(None, seq_len)

            if examples_plotted >= num_examples:
                break



            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1.2])

            # ====== (1) Attention + Feature Values Overlay ======
            attn_weights  = attn_weights_np[i, padding_slice]
            ax_att = fig.add_subplot(gs[0, 0])
            ax_att.scatter(np.arange(seq_len), attn_weights, color='k', marker='o', label='Attention')
            ax_att.set_ylabel("Attention")

            ax_feat = fig.add_subplot(gs[1, 0])
            feats = feature_values[i, padding_slice, :]
            for f_idx in range(feats.shape[1]):
                ax_feat.plot(np.arange(seq_len), feats[:, f_idx], label=(features[f_idx] if features else f"F{f_idx}"), alpha=0.7)

            ax_feat.set_xlabel("Time Step")
            ax_feat.set_ylabel("Covariates")
            ax_feat.legend(fontsize=8, ncol=2)
            ax_feat.grid(True)

            # ====== (2) Predicted Survival Distribution ======
            ax_sd = fig.add_subplot(gs[:, 1])  # span both rows
            abs_bin_edges = bin_edges + seq_len
            bin_widths = np.diff(abs_bin_edges)
            bin_widths[-1] = bin_widths[-2]  # make last bin same width for visualization

            ax_sd.bar(
                abs_bin_edges[:-1],
                predicted_pmf[i].cpu().numpy(),
                width=bin_widths,
                align='edge',
                color='tab:blue',
                edgecolor='k',
                alpha=0.5
            )

            ax_sd.bar(
                abs_bin_edges[:-1],
                pmfs[i],
                width=bin_widths,
                align='edge',
                color='tab:red',
                edgecolor='k',
                alpha=0.5,
                label='True PMF')

            abs_event_time = time_to_events[i] + seq_len
            if event_indicators[i] == 1:
                ax_sd.axvline(abs_event_time, color='red', linestyle='--', label="True Event")
            else:
                ax_sd.axvline(abs_event_time, color='orange', linestyle='--', label="Censored")

            ax_sd.set_title(f"Predicted Survival Distribution {cell_idxs[i]}, {files[i]}")
            ax_sd.set_xlabel("Absolute Time")
            ax_sd.set_ylabel("Probability")
            ax_sd.legend()
            ax_sd.grid(True)

            # ====== SAVE OR SHOW ======
            if save_path:
                out = save_path / f"val_pred_cell_{examples_plotted+1}.png"
                plt.savefig(out, dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

            examples_plotted += 1

        if examples_plotted >= num_examples:
            break
        
def eval_model(model_, dataloader, save_dir: Path, device: str) -> None:
    for batch in dataloader:
        outputs = model(batch['features'], batch['length'])
        
        true_bins = batch['time_to_event_bin']
        
        plot_cm(outputs, true_bins, save_dir / 'cm.png')
        

def plot_cm(outputs, true_bins, save_as) -> None:
    pass
        
        
       
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model_, dataloader, device, num_bins, save_path=None):
    """
    Plot a confusion matrix comparing predicted vs true event bins.
    Predicted bin is argmax over predicted PMF.
    """
    model_.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Getting confusion matrix data'):
            outputs = model_(batch['features'], batch['length'])
            predicted_pmf = model.estimated_pmf(outputs)

            # True bins
            true_bins = batch['time_to_event_bin'].cpu().numpy()
            events = batch['event_indicator'].cpu().numpy()

            # Predicted bins (argmax over PMF)
            pred_bins = predicted_pmf.argmax(dim=1).cpu().numpy()

            # Only include uncensored events
            all_true.extend(true_bins[events == 1])
            all_pred.extend(pred_bins[events == 1])


    print(np.unique(all_true, return_counts=True), np.unique(all_pred, return_counts=True))
    # Compute confusion matrix
    cm = confusion_matrix(all_true, all_pred, labels=np.arange(num_bins))
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=True)
    plt.xlabel("Predicted Time Bin")
    plt.ylabel("True Time Bin")
    plt.title("Confusion Matrix of Predicted vs True Time-to-Event Bins")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_time_calibration(
    model_,
    dataloader,
    device,
    num_bins,
    num_prob_bins=10,
    save_path=None
):
    """
    Plot a global calibration curve for predicted time-bin probabilities.
    Aggregates over all samples and all time bins.
    """
    model_.eval()

    all_pred_probs = []
    all_outcomes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting calibration data"):
            outputs = model_(
                batch['features'],
                batch['length'],
            )

            predicted_pmf = model.estimated_pmf(outputs)  # (B, num_bins)

            true_bins = batch['time_to_event_bin']          # (B,)
            events = batch['event_indicator']               # (B,)

            # Only uncensored events
            mask = events == 1

            pmf = predicted_pmf[mask]        # (N, num_bins)
            true_bins = true_bins[mask]      # (N,)

            # For every sample and every bin:
            for i in range(pmf.shape[0]):
                for k in range(num_bins):
                    all_pred_probs.append(pmf[i, k].item())
                    all_outcomes.append(
                        1.0 if k == true_bins[i].item() else 0.0
                    )

    all_pred_probs = np.array(all_pred_probs)
    all_outcomes = np.array(all_outcomes)

    # Bin by predicted probability
    prob_bins = np.linspace(0.0, 1.0, num_prob_bins + 1)
    bin_ids = np.digitize(all_pred_probs, prob_bins) - 1

    mean_pred = []
    empirical_freq = []

    for b in range(num_prob_bins):
        idx = bin_ids == b
        if idx.sum() == 0:
            continue

        mean_pred.append(all_pred_probs[idx].mean())
        empirical_freq.append(all_outcomes[idx].mean())

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, empirical_freq, marker='o', label="Model")
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")

    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title("Calibration Plot for Time-to-Event Distributions")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_soft_confusion_matrix(
    model_,
    dataloader,
    device,
    num_bins,
    save_path=None
):
    """
    Plot a soft confusion matrix using the full predicted PMF
    instead of argmax predictions.
    """
    model_.eval()

    soft_cm = np.zeros((num_bins, num_bins), dtype=np.float64)
    row_counts = np.zeros(num_bins, dtype=np.float64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting soft confusion data"):
            outputs = model_(
                batch['features'],
                batch['length'],
            )

            predicted_pmf = model.estimated_pmf(outputs)  # (B, num_bins)

            true_bins = batch['time_to_event_bin'].cpu().numpy()
            events = batch['event_indicator'].cpu().numpy()

            pmf = predicted_pmf.cpu().numpy()

            for i in range(len(true_bins)):
                if events[i] != 1:
                    continue

                t = true_bins[i]
                soft_cm[t] += pmf[i]       # add full distribution
                row_counts[t] += 1

    # Normalize per true bin
    soft_cm = soft_cm / np.maximum(row_counts[:, None], 1e-8)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        soft_cm,
        cmap="Blues",
        cbar=True
    )

    plt.xlabel("Predicted Time Bin")
    plt.ylabel("True Time Bin")
    plt.title("Soft Confusion Matrix (Average Predicted Distributions)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def average_feature_importance(model, dataloader, device, features, save_as: Path):
    """
    Compute average gradient-based feature importance, aligned by event time.
    """
    model.eval()
    all_importances = []
    lengths_list = []

    for batch in tqdm(dataloader, desc="Computing gradient feature importance"):
        # cell_features, lengths, time_to_event_bins, event_indicators, time_to_events, cell_idxs, files, pmfs = batch
        # cell_features = cell_features.to(device)
        # lengths_np = lengths.cpu().numpy()
        # time_to_events_np = time_to_events.cpu().numpy()
        # event_indicators_np = event_indicators.cpu().numpy()

        lengths = batch['length'].cpu().numpy()
        pmfs = batch['binned_pmf']
        time_to_events = batch['time_to_event'].cpu().numpy()
        event_indicators = batch['event_indicator'].cpu().numpy()
        cell_idxs = batch['cell_idx']
        files = batch['hdf5_path']  
        cell_features = batch['features'].to(device)
        
        batch_size = cell_features.shape[0]

        for i in range(batch_size):
            if event_indicators[i] != 1:  # only align uncensored sequences
                continue

            # seq_len = lengths_np[i]
            # x = cell_features[i, :seq_len, :].detach().clone()
            x = cell_features[i].detach().clone()
            x.requires_grad_(True)

            model_was_training = model.training
            model.train()
            pred, _ = model(x.unsqueeze(0), batch['length'][i].unsqueeze(0), return_attention=True)
            scalar = pred.sum()  # sum over bins
            model.zero_grad()
            scalar.backward()

            grads = x.grad.detach().cpu().numpy()  # (seq_len, num_features)
            importance = np.abs(grads)
            # importance=grads
            # print(importance.shape)

            # Append NaNs for timesteps after event if needed
            time_to_event = time_to_events[i]
            aligned_imp = np.concatenate([importance, np.full((int(time_to_event), importance.shape[1]), np.nan)])

            all_importances.append(aligned_imp)
            lengths_list.append(aligned_imp.shape[0])

            if not model_was_training:
                model.eval()

    max_len = max(lengths_list)
    # Prepend NaNs so events align at the same index
    padded_importances = [
        np.concatenate([np.full((max_len - imp.shape[0], imp.shape[1]), np.nan), imp])
        for imp in all_importances
    ]

    stacked = np.stack(padded_importances, axis=0)  # (num_cells, num_frames, num_features)
    # avg_importance = np.nanmean(stacked, axis=0)
    avg_importance = np.nanpercentile(stacked, 50, axis=0)
    lower = np.nanpercentile(stacked, 25, axis=0)
    upper = np.nanpercentile(stacked, 75, axis=0)

    # Plot each feature as a line with confidence interval
    plt.figure(figsize=(12,6))
    x = np.arange(-max_len, 0)  # negative steps leading to event
    for f_idx, feat_name in enumerate(features):
        plt.plot(x, avg_importance[:, f_idx], label=feat_name)
        plt.fill_between(x, lower[:, f_idx], upper[:, f_idx], alpha=0.2)
    plt.xlabel("Time Step (aligned to event)")
    plt.ylabel("Gradient-based Feature Importance")
    plt.title("Average Feature Importance Across Validation Set TEST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

def CAM(model_: torch.nn.Module,
        target_layer: torch.nn.Module,
        x: torch.Tensor,
        lengths: torch.Tensor):
    """
    Compute Grad-CAM for temporal CNNs.

    Args:
        model: the temporal CNN model
        target_layer: the layer to compute CAM for
        x: (B, T, C) input sequence
        lengths: (B,) sequence lengths
        target_scalar: scalar output to backprop
    Returns:
        cam_map: (B, T) importance over time
    """ 
    cam = model.TemporalGradCAM(model_, target_layer)
    outputs = model_(x, lengths)
    
    target_bin = outputs.argmax(dim=1)
    target = outputs[torch.arange(outputs.size(0)), target_bin].sum()
    cam_map = cam(x, lengths, target)
    return cam_map

def smoothgrad_saliency_single(model, x, length, n_samples=1, sigma_ratio=0.1, mask_padded=True):
    """
    SmoothGrad saliency for a single sequence (B=1)
    x: (T, C)
    length: scalar
    """
    model.eval()
    x = x.unsqueeze(0)  # (1, T, C)
    length = length.unsqueeze(0)
    B, T, C = x.shape
    saliency_accum = torch.zeros_like(x)

    # Forward to get target bin
    with torch.no_grad():
        outputs = model(x, length)  # (1, num_bins)
    target_bin = int(outputs.argmax(dim=1).item())

    for _ in range(n_samples):
        sigma = sigma_ratio * (outputs.max() - outputs.min())
        noise = torch.randn_like(x) * sigma
        x_noisy = (x + noise).detach().clone().requires_grad_(True)

        pred = model(x_noisy, length)
        scalar = pred[0, target_bin].sum()

        model.zero_grad()
        scalar.backward()
        saliency_accum += x_noisy.grad.detach()

    saliency_map = saliency_accum / n_samples

    if mask_padded:
        mask = torch.arange(T, device=length.device).unsqueeze(0) >= (T - length).unsqueeze(1)
        mask = ~mask
        saliency_map *= mask.unsqueeze(-1)

    return saliency_map[0].cpu().numpy()  # (T, C)

    
def validate(model_, model_dir, val_hdf5_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(model_dir / 'model_params.json', 'r') as f:
        model_params = json.load(f)

    features = model_params['features']
    model_ = model_(**model_params)
    model_.load_state_dict(torch.load(model_dir / 'model.pth', map_location=device))
    model_.to(device)

    normalisation_means = model_params['normalization_means']
    normalisation_stds = model_params['normalization_stds']
    validate_dataset = CellDataset(
        hdf5_paths=val_hdf5_paths,
        features=features,
        means=np.array(normalisation_means),
        stds=np.array(normalisation_stds),
        num_bins=model_params['output_size'],
        event_time_bins=np.array(model_params['event_time_bins']),
        uncensored_only=True,
        min_length=100,
        max_time_to_death=200,
    )

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=256, shuffle=False, collate_fn= lambda x: collate_fn(x, dataset=validate_dataset, device=device))

    loss_fn = model.compute_loss

    # Plot confusion matrices and calibration
    plot_soft_confusion_matrix(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'soft_confusion_matrix.png')
    plot_time_calibration(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'time_calibration.png')
    plot_confusion_matrix(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'confusion_matrix.png')

    val_loss = validate_step(model_, validate_loader, loss_fn, device)
    print(f"Validation Loss: {val_loss}")
    # average_attention(model_, validate_loader, device, model_dir / 'attention_weights.jpeg')
    # average_feature_importance(model_, validate_loader, device, features, model_dir / 'feature_importance.jpeg')
    visualize_validation_predictions(model_, validate_loader, device, num_examples=50, save_path=model_dir, bin_edges=np.array(model_params['event_time_bins']), features=features, padding_at='start')


def main():
    # datasets = list((Path('PhagoPred')/'Datasets'/ 'ExposureTest').iterdir())
    # train_datasets = datasets[:-2]
    # val_datasets = datasets[-2:]
    val_datasets = [Path('PhagoPred')/'Datasets'/'val_synthetic.h5']
    validate(
        model_=model.TemporalCNN,
        model_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'temp_cnn' / 'test_run',
        val_hdf5_paths=val_datasets,
    )
    
if __name__ == "__main__":
    main()