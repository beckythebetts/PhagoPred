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
    model, 
    dataset, 
    device, 
    bin_edges, 
    features=None,               
    num_examples=20, 
    save_path=None
):

    model.eval()
    model.to(device)
    examples_plotted = 0
    pbar = tqdm(total=num_examples, desc="Visualizing validation predictions")

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, dataset=dataset, device=device)
    )
    
    for batch in dataloader:
        
        fig = plt.figure(figsize=(12, 6))
        
        saliency_map_ = saliency_map(model, batch['features'], batch['length'])
        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(
            saliency_map_.T,
            aspect='auto',
            cmap='viridis',
            extent=[0, saliency_map_.shape[0], 0, saliency_map_.shape[1]],
            origin='lower'
        )
        ax.set_title('Saliency Map')
        ax.set_ylabel('Features')
        if features is not None:
            ax.set_yticks(np.arange(len(features))+0.5)
            ax.set_yticklabels(features)
        ax_colorbar = plt.colorbar(ax.images[0], ax=ax)
        ax_colorbar.set_label('Saliency')   
        plt.savefig(save_path / f"val_pred_cell_{examples_plotted+1}.png")
        plt.close(fig)
        
        examples_plotted += 1
        pbar.update(1)
        if examples_plotted >= num_examples:
            break
    pbar.close()
        # output, attn = model(batch['features'], batch['length'], return_attention=True)
        
def eval_model(model_, dataloader, save_dir: Path, device: torch.Device) -> None:
    for batch in dataloader:
        outputs = model(batch['features'], batch['length'])
        
        true_bins = batch['time_to_event_bin']
        
        plot_cm(outputs, true_bins, save_dir / 'cm.png')
        

def plot_cm(outputs, true_bins, save_as) -> None:
    pass
        
        
       
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
    # average_attention(model_, validate_loader, device, model_dir / 'attention_weights.jpeg')
    # average_feature_importance(model_, validate_loader, device, features, model_dir / 'feature_importance.jpeg')
    visualize_validation_predictions(model_, validate_dataset, device, num_examples=50, save_path=model_dir, bin_edges=np.array(model_params['event_time_bins']), features=features)


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