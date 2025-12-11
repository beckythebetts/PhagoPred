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

def visualize_validation_predictions(
    model, 
    dataloader, 
    device, 
    bin_edges, 
    features=None,               # ← OPTIONAL: list of feature names
    num_examples=20, 
    save_path=None
):
    import matplotlib.pyplot as plt

    model.eval()
    model.to(device)
    examples_plotted = 0

    for batch in dataloader:
        lengths = batch['length'].cpu().numpy()
        pmfs = batch['binned_pmf']
        time_to_events = batch['time_to_event'].cpu().numpy()
        event_indicators = batch['event_indicator'].cpu().numpy()
        cell_idxs = batch['cell_idx']
        files = batch['hdf5_path']  
        
              
        # with torch.no_grad():
        #     predicted_dists, attn_weights = model(batch['features'], batch['length'], return_attention=True)

        # predicted_dists_np = predicted_dists.cpu().numpy()
        # attn_weights_np = attn_weights.cpu().numpy()

        for i in range(len(batch['features'])):
            if examples_plotted >= num_examples:
                break

            seq_len = lengths[i]

            # --------------------------------------------------
            #  Feature gradients (optional, still computed)
            # --------------------------------------------------
            x = batch['features'][i, -seq_len:, :].detach().clone().to(device)
            length = batch['length'][i].detach().clone().to(device)
            x.requires_grad_(True)

            model_was_training = model.training
            model.train()
            pred_single, attn_single = model(x.unsqueeze(0), length.unsqueeze(0), return_attention=True)
            scalar = pred_single.sum()
            model.zero_grad()
            scalar.backward()
        

            attn_vec = attn_single[0, -seq_len:].detach().cpu().numpy()  # (seq_len,)
            # attn_vec = attn_single[0].detach().cpu().numpy()
            feature_values = x.detach().cpu().numpy()  # (seq_len, num_features)
            
            # target_layer = model.cn_layers[-1]  # last conv layer
            # cam_map = CAM(model, target_layer, batch['features'], batch['length'])
            # cam_vec = cam_map[i, -seq_len:].detach().cpu().numpy() 
            
            cam_maps = [CAM(model, layer, batch['features'], batch['length']) for layer in model.cn_layers]
            cam_map = torch.stack(cam_maps, dim=0).mean(dim=0)
            cam_vec = cam_map[i, -seq_len:].detach().cpu().numpy() 
            
            saliency_map = smoothgrad_saliency_single(model, x, length, n_samples=20, sigma_ratio=0.1, mask_padded=False)


            if not model_was_training:
                model.eval()

            # --------------------------------------------------
            #  PLOTTING
            # --------------------------------------------------
            fig = plt.figure(figsize=(14, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1.2])

            # ====== (1) Attention + Feature Values Overlay ======
            ax_att = fig.add_subplot(gs[0, 0])
            # ax_att.scatter(np.arange(seq_len), attn_vec, color='k', marker='o', label='Attention')
            # ax_att.set_ylabel("Attention")
            
            # ax_att.scatter(np.arange(len(attn_vec)), attn_vec, color='k', marker='o', label='Attention')
            # ax_att.plot(np.arange(len(cam_vec)), cam_vec, color='tab:green', label='Grad-CAM')
            print(np.unique(saliency_map))
            saliency_avg = saliency_map.mean(axis=1)
            ax_att.plot(np.arange(len(saliency_avg)), saliency_avg, color='tab:red', label='SmoothGrad')
            ax_att.set_ylabel("Attention / CAM / Saliency")
            ax_att.set_ylabel("Attention / CAM")
            ax_att.legend(fontsize=8)

            # Optionally normalize features for plotting on same axis
            # feat_norm = (feature_values - feature_values.min(0)) / (feature_values.max(0) - feature_values.min(0) + 1e-6)
            
            ax_feat = fig.add_subplot(gs[1, 0])
            for f_idx in range(feature_values.shape[1]):
                ax_feat.plot(np.arange(seq_len), feature_values[:, f_idx], label=(features[f_idx] if features else f"F{f_idx}"), alpha=0.7)

            # ax_att_feat.set_title("Attention (black) + Feature Values (colored)")
            ax_feat.set_xlabel("Time Step")
            ax_feat.set_ylabel("Covariates")
            ax_feat.legend(fontsize=8, ncol=2)
            ax_feat.grid(True)

            # ====== (2) Predicted Survival Distribution ======
            ax_sd = fig.add_subplot(gs[:, 1])  
            abs_bin_edges = bin_edges + seq_len
            bin_widths = np.diff(abs_bin_edges)
            bin_widths[-1] = 100.0  # avoid zero width

            # pmf = np.append(np.zeros(seq_len), pmfs[i])
            
            ax_sd.bar(
                abs_bin_edges[:-1],
                pred_single[0].detach().cpu().numpy(),
                width=bin_widths,
                align='edge',
                color='tab:blue',
                edgecolor='k',
                alpha=0.5
            )
            
            # ax_sd.fill_between(
            #     np.arange(len(pmf)),
            #     0,
            #     pmf,
            #     color='r',
            #     alpha=0.5,
            #     label='True time to event probability distribution'
            # )
            
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
            
            _, attn_weights = model(cell_features, batch['length'], return_attention=True)
            attn_weights_np = attn_weights.cpu().numpy()  # (batch, seq_len)

            for i in range(len(attn_weights_np)):
                if event_indicators[i] == 1:
                    attn_vec = attn_weights_np[i, -lengths[i]:]
                    time_to_event = time_to_events[i]
                    aligned_vec = np.concatenate([attn_vec, np.full(int(time_to_event), np.nan)])

                    attn_accum.append(aligned_vec)
                    lengths_list.append(len(aligned_vec))
        max_len = max(lengths_list)
        padded_attn = [np.concatenate([np.full(max_len - len(attn), np.nan), attn]) for attn in attn_accum]
        
    attn = np.stack(padded_attn, axis=0)
    # avg_attn = np.nanmean(attn, axis=0)
    avg_attn = np.nanpercentile(attn, 50, axis=0)
    lower = np.nanpercentile(attn, 45, axis=0)
    upper = np.nanpercentile(attn, 65, axis=0)
    
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

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=32, shuffle=False, collate_fn= lambda x: collate_fn(x, dataset=validate_dataset, device=device))

    loss_fn = model.compute_loss

    val_loss = validate_step(model_, validate_loader, loss_fn, device)
    # print(f"Validation Loss: {val_loss:.4f}")
    average_attention(model_, validate_loader, device, model_dir / 'attention_weights.jpeg')
    average_feature_importance(model_, validate_loader, device, features, model_dir / 'feature_importance.jpeg')
    visualize_validation_predictions(model_, validate_loader, device, num_examples=50, save_path=model_dir, bin_edges=np.array(model_params['event_time_bins']), features=features)


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