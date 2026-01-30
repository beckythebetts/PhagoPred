from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import json
import numpy as np

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models.dynamic_deephit import model

def validate_step(model_, dataloader, loss_fn, device):
    model_.eval()
    # total_loss = 0.0
    losses = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            # cell_features, lengths, time_to_event_bins, event_indicators, time_to_events, cell_idxs, files, pmfs = batch
            # cell_features = cell_features.to(device)
            # lengths = lengths.to(device)
            # time_to_event_bins = time_to_event_bins.to(device)
            # event_indicators = event_indicators.to(device)
            # time_to_events = time_to_events.to(device)

            outputs, y = model_(batch['features'], batch['length'], mask=batch['mask'])
            loss_values = loss_fn(outputs, batch['time_to_event_bin'], batch['event_indicator'], batch['features'], y, mask=batch['mask'])
            for key, value in zip(
                ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Prediction Loss', 'Censored Loss', 'Uncensored Loss'], loss_values):
                losses[key] += value.item() * batch['features'].size(0)  # Multiply by batch size

    avg_losses = {key: value / len(dataloader.dataset) for key, value in losses.items()}
    return avg_losses

def visualize_validation_predictions(
    model_, 
    dataloader, 
    device, 
    bin_edges, 
    features=None,               # ← OPTIONAL: list of feature names
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
        # pmfs = batch['pmf']
        pmfs = batch['binned_pmf']
        time_to_events = batch['time_to_event'].cpu().numpy()
        event_indicators = batch['event_indicator'].cpu().numpy()
        cell_idxs = batch['cell_idx']
        files = batch['hdf5_path']  
        feature_values = batch['features'].cpu().numpy()
        
              
        with torch.no_grad():
            outputs, _, attn_weights = model_(batch['features'], batch['length'], return_attention=True, mask=batch['mask'])
        
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
            # print(attn_weights_np.shape, feature_values.shape)
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
                # np.arange(len(pmfs[i])),
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

def average_attention(model, dataloader, device, save_as: Path, max_len=None):
    """
    Compute average attention weights over the validation set.
    
    alignment: 'start', 'end', 'event'
    max_len: optional truncation/padding for averaging
    """
    model.eval()
    attn_accum = []
    lengths_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Getting average attention scores'):
            cell_features, lengths, time_to_event_bins, event_indicators, time_to_events, cell_idxs, files, pmfs = batch
            cell_features = cell_features.to(device)
            lengths_np = lengths.cpu().numpy()
            time_to_events_np = time_to_events.cpu().numpy()
            event_indicators_np = event_indicators.cpu().numpy()

            _, _, attn_weights = model(cell_features, return_attention=True)
            attn_weights_np = attn_weights.cpu().numpy()  # (batch, seq_len)

            for i in range(len(attn_weights_np)):
                if event_indicators_np[i] == 1:
                    attn_vec = attn_weights_np[i, :lengths_np[i]]
                    time_to_event = time_to_events_np[i]
                    aligned_vec = np.concatenate([attn_vec, np.full(time_to_event, np.nan)])

                    attn_accum.append(aligned_vec)
                    lengths_list.append(len(aligned_vec))
        max_len = max(lengths_list)
        padded_attn = [np.concatenate([np.full(max_len - len(attn), np.nan), attn]) for attn in attn_accum]
        
    attn = np.stack(padded_attn, axis=0)
    # avg_attn = np.nanmean(attn, axis=0)
    avg_attn = np.nanpercentile(attn, 50, axis=0)
    lower = np.nanpercentile(attn, 25, axis=0)
    upper = np.nanpercentile(attn, 75, axis=0)
    
    plt.figure(figsize=(12,4))
    plt.plot(np.arange(-len(avg_attn), 0), avg_attn, color='k')
    plt.fill_between(np.arange(-len(avg_attn), 0), lower, upper, color='k', alpha=0.2, edgecolor='none')
    plt.xlabel("Time Step")
    plt.ylabel("Average Attention Weight")
    plt.title("Average Attention Weights Across Validation Set")
    plt.grid()
    plt.savefig(save_as)
    plt.close()

def average_feature_importance(model, dataloader, device, features, save_as: Path):
    """
    Compute average gradient-based feature importance, aligned by event time.
    """
    model.eval()
    all_importances = []
    lengths_list = []

    for batch in tqdm(dataloader, desc="Computing gradient feature importance"):
        cell_features, lengths, time_to_event_bins, event_indicators, time_to_events, cell_idxs, files, pmfs = batch
        cell_features = cell_features.to(device)
        lengths_np = lengths.cpu().numpy()
        time_to_events_np = time_to_events.cpu().numpy()
        event_indicators_np = event_indicators.cpu().numpy()

        batch_size = cell_features.shape[0]

        for i in range(batch_size):
            if event_indicators_np[i] != 1:  # only align uncensored sequences
                continue

            # seq_len = lengths_np[i]
            # x = cell_features[i, :seq_len, :].detach().clone()
            x = cell_features[i].detach().clone()
            x.requires_grad_(True)

            model_was_training = model.training
            model.train()
            pred, _, _ = model(x.unsqueeze(0), return_attention=True)
            scalar = pred.sum()  # sum over bins
            model.zero_grad()
            scalar.backward()

            grads = x.grad.detach().cpu().numpy()  # (seq_len, num_features)
            importance = np.abs(grads)
            # print(importance.shape)

            # Append NaNs for timesteps after event if needed
            time_to_event = time_to_events_np[i]
            aligned_imp = np.concatenate([importance, np.full((time_to_event, importance.shape[1]), np.nan)])

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
    # mean_importance = np.nanmean(stacked, axis=0)
    median_importance = np.nanpercentile(stacked, 50, axis=0)
    lower = np.nanpercentile(stacked, 5, axis=0)
    upper = np.nanpercentile(stacked, 95, axis=0)

    # Plot each feature as a line with confidence interval
    plt.figure(figsize=(12,6))
    x = np.arange(-max_len, 0)  # negative steps leading to event
    for f_idx, feat_name in enumerate(features):
        plt.plot(x, median_importance[:, f_idx], label=feat_name)
        plt.fill_between(x, lower[:, f_idx], upper[:, f_idx], alpha=0.2)
    plt.xlabel("Time Step (aligned to event)")
    plt.ylabel("Gradient-based Feature Importance")
    plt.title("Average Feature Importance Across Validation Set TEST")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

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
        for batch in tqdm(dataloader, desc='GEtting confusion matrix data'):
            outputs, _ = model_(batch['features'], batch['length'], mask=batch['mask'])
            predicted_pmf = model.estimated_pmf(outputs)

            # True bins
            true_bins = batch['time_to_event_bin'].cpu().numpy()
            events = batch['event_indicator'].cpu().numpy()

            # Predicted bins (argmax over PMF)
            pred_bins = predicted_pmf.argmax(dim=1).cpu().numpy()
            print(true_bins, pred_bins)

            # Only include uncensored events (optional)
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
            outputs, _ = model_(
                batch['features'],
                batch['length'],
                mask=batch['mask'],
            )

            predicted_pmf = model.estimated_pmf(outputs)  # (B, num_bins)

            true_bins = batch['time_to_event_bin']          # (B,)
            events = batch['event_indicator']               # (B,)

            # Only uncensored events (same choice as your confusion matrix)
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
            outputs, _ = model_(
                batch['features'],
                batch['length'],
                mask=batch['mask']
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
        min_length=500,
        max_time_to_death=100,
    )
    
    print(len(validate_dataset), "validation samples loaded.")

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=256, shuffle=False, collate_fn= lambda x: collate_fn(x, dataset=validate_dataset, device=device, pad_at='end'))
    plot_soft_confusion_matrix(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'soft_confusion_matrix.png')
    plot_time_calibration(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'time_calibration.png')
    plot_confusion_matrix(model_, validate_loader, device, num_bins=model_params['output_size'], save_path=model_dir / 'confusion_matrix.png')
    loss_fn = model.compute_loss

    val_loss = validate_step(model_, validate_loader, loss_fn, device)
    # print(f"Validation Loss: {val_loss:.4f}")
    # average_attention(model_, validate_loader, device, model_dir / 'attention_weights.jpeg')
    # average_feature_importance(model_, validate_loader, device, features, model_dir / 'feature_importance.jpeg')
    visualize_validation_predictions(model_, validate_loader, device, num_examples=50, save_path=model_dir, bin_edges=np.array(model_params['event_time_bins']), features=features, padding_at='end')

def main():
    # datasets = list((Path('PhagoPred')/'Datasets'/ 'ExposureTest').iterdir())
    # train_datasets = datasets[:-2]
    # val_datasets = datasets[-2:]
    val_datasets = [Path('PhagoPred')/'Datasets'/'val_synthetic.h5']
    validate(
        model_=model.DynamicDeepHit,
        model_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'dynamic_deephit' / 'test_run',
        val_hdf5_paths=val_datasets,
    )
    
if __name__ == "__main__":
    main()