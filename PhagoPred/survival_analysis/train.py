from pathlib import Path
import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models import dynamic_deephit
from PhagoPred.survival_analysis.validate import validate_step, visualize_validation_predictions


def train_step(model, dataloader, optimiser, loss_fn, device):
    model.train()
    # losses = {'Total Loss': 0.0, 'NLL Loss': 0.0, 'Ranking Loss': 0.0, 'Prediction Loss': 0.0}
    # total_loss = 0.0
    losses = defaultdict(float)
    
    for batch in dataloader:
        cell_features, lengths, time_to_event_bins, event_indicators, time_to_event = batch
        cell_features = cell_features.to(device)
        lengths = lengths.to(device)
        time_to_event_bins = time_to_event_bins.to(device)
        event_indicators = event_indicators.to(device)

        optimiser.zero_grad()
        outputs, y = model(cell_features)
        loss_values = loss_fn(outputs, cell_features, y, time_to_event_bins, event_indicators)
        
        loss = loss_values[0]
        loss.backward()
        optimiser.step()
        
        for key, value in zip(
            ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Prediction Loss'], loss_values
        ):
            losses[key] += value.item() * cell_features.size(0)

    avg_losses = {key: value / len(dataloader.dataset) for key, value in losses.items()}
    return avg_losses



def train(model, model_params, train_hdf5_paths: list, val_hdf5_paths: list, features: list, optimiser, loss_fn, num_epochs, save_dir: Path, batch_size: int, lr: float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model_params = {k: v.to()}
    model = model(**model_params)
    model = model.to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_dataset = CellDataset(
        hdf5_paths=train_hdf5_paths,
        features=features,
        num_bins=model_params['output_size'],
    )

    bins = train_dataset.get_bins()
    train_dataset.plot_event_vs_censoring_hist(save_path=save_dir / 'train_event_censoring_histogram.png', title='Training Set Event vs Censoring Histogram')
    normalisation_means, normalization_stds = train_dataset.get_normalization_stats()
    model_params['normalization_means'] = normalisation_means.tolist()
    model_params['normalization_stds'] = normalization_stds.tolist()
    model_params['event_time_bins'] = bins.tolist()
    normalisation_means = torch.tensor(normalisation_means, dtype=torch.float32)
    normalization_stds = torch.tensor(normalization_stds, dtype=torch.float32)

    validate_dataset = CellDataset(
        hdf5_paths=val_hdf5_paths,
        features=features,
        means=normalisation_means,
        stds=normalization_stds,
        num_bins=model_params['output_size'],
        event_time_bins=bins,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, means=normalisation_means, stds=normalization_stds, device=device), num_workers=0)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, means=normalisation_means, stds=normalization_stds, device=device), num_workers=0)

    with open(save_dir / 'model_params.json', 'w') as f:
        json.dump(model_params, f)
        
    optimiser = optimiser(model.parameters(), lr=lr)
    training_json = save_dir / 'training.jsonl'
    
    training_json  = save_dir / 'training.jsonl'
    with open(training_json, 'w') as f:
            for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
                train_losses = train_step(model, train_loader, optimiser, loss_fn, device)
                validate_losses = validate_step(model, validate_loader, loss_fn, device)
                losses_dict = {'epoch': epoch, 'train': train_losses, 'validate': validate_losses}
                print(losses_dict)
                f.write(json.dumps(losses_dict) + '\n')

    # --- Step 7: Save model and plot ---
    torch.save(model.state_dict(), save_dir / 'model.pth')
    plot_training_losses(training_json, save_dir / 'loss_plot.png')
    print(f"Training complete. Model saved to {save_dir}")

    visualize_validation_predictions(model, validate_loader, device, num_examples=5, save_path=save_dir , bin_edges=bins)

    

def train_single_dataset(
    model,
    model_params,
    hdf5_path: Path,
    features: list,
    optimiser,
    loss_fn,
    num_epochs: int,
    save_dir: Path,
    batch_size: int,
    lr: float,
    val_split: float = 0.2,
    seed: int = 42,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = model(**model_params)
    model = model.to(device)
    
    os.makedirs(save_dir, exist_ok=True)

    # --- Step 1: Load metadata to get available cell indices ---
    dummy_dataset = CellDataset(hdf5_paths=[hdf5_path], features=features)
    all_idxs = np.arange(len(dummy_dataset))  # Total number of cells in the file

    # --- Step 2: Split into train and validation sets ---
    train_idxs, val_idxs = train_test_split(
        all_idxs,
        test_size=val_split,
        random_state=seed,
        shuffle=True
    )

    # --- Step 3: Initialize datasets with selected indices ---
    train_dataset = CellDataset(
        hdf5_paths=[hdf5_path],
        features=features,
        specified_cell_idxs=train_idxs.tolist(),
        num_bins=model_params['output_size']
    )
    train_dataset.plot_event_vs_censoring_hist(save_path=save_dir / 'train_event_censoring_histogram.png', title='Training Set Event vs Censoring Histogram')
    bins = train_dataset.get_bins()

    normalisation_means, normalization_stds = train_dataset.get_normalization_stats()
    model_params['normalization_means'] = normalisation_means.tolist()
    model_params['normalization_stds'] = normalization_stds.tolist()
    model_params['event_time_bins'] = bins.tolist()
    normalisation_means = torch.tensor(normalisation_means, dtype=torch.float32)
    normalization_stds = torch.tensor(normalization_stds, dtype=torch.float32)

    validate_dataset = CellDataset(
        hdf5_paths=[hdf5_path],
        features=features,
        specified_cell_idxs=val_idxs.tolist(),
        num_bins =model_params['output_size'],
        event_time_bins=bins,
    )
    validate_dataset.plot_event_vs_censoring_hist(save_path=save_dir / 'val_event_censoring_histogram.png', title='Validation Set Event vs Censoring Histogram')

    # --- Step 4: Create data loaders ---
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, means=normalisation_means, stds=normalization_stds, device=device), num_workers=4, pin_memory=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, means=normalisation_means, stds=normalization_stds, device=device), num_workers=4, pin_memory=True)

    # # --- Step 5: Save model params ---
    with open(save_dir / 'model_params.json', 'w') as f:
        json.dump(model_params, f)

    # --- Step 6: Train loop ---
    optimiser = optimiser(model.parameters(), lr=lr)
    training_json = save_dir / 'training.jsonl'

    with open(training_json, 'w') as f:
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
            train_losses = train_step(model, train_loader, optimiser, loss_fn, device)
            validate_losses = validate_step(model, validate_loader, loss_fn, device)
            losses_dict = {'epoch': epoch, 'train': train_losses, 'validate': validate_losses}
            print(losses_dict)
            f.write(json.dumps(losses_dict) + '\n')

    # --- Step 7: Save model and plot ---
    torch.save(model.state_dict(), save_dir / 'model.pth')
    plot_training_losses(training_json, save_dir / 'loss_plot.png')
    print(f"Training complete. Model saved to {save_dir}")

    visualize_validation_predictions(model, validate_loader, device, num_examples=5, save_path=save_dir , bin_edges=bins)


def plot_training_losses(losses_json_path: Path, output_path: Path = None):
    # Load losses dictionary from JSON
    all_losses = {}
    with open(losses_json_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            epoch = str(entry['epoch'])
            all_losses[epoch] = {
                'train': entry['train'],
                'validate': entry['validate']
            }

    # Extract epochs sorted
    epochs = sorted(int(e) for e in all_losses.keys())

    # Get all loss types from the first epoch's train dict keys
    loss_types = list(all_losses[str(epochs[0])]['train'].keys())

    # Prepare dicts to hold losses per type
    train_losses = {loss_type: [] for loss_type in loss_types}
    val_losses = {loss_type: [] for loss_type in loss_types}

    # Fill in losses per epoch
    for epoch in epochs:
        epoch_str = str(epoch)
        for loss_type in loss_types:
            train_losses[loss_type].append(all_losses[epoch_str]['train'][loss_type])
            val_losses[loss_type].append(all_losses[epoch_str]['validate'][loss_type])
            

    plt.figure(figsize=(12, 8))

    cmap = plt.get_cmap('tab10')
    # Plot each loss type: train solid line, val dashed line
    for i, loss_type in enumerate(loss_types):
        plt.plot(epochs, train_losses[loss_type], label=f'Train {loss_type}', linestyle='-', color=cmap(i))
        plt.plot(epochs, val_losses[loss_type], label=f'Validation {loss_type}', linestyle='--', color=cmap(i))

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses Over Epochs')
    plt.legend()
    plt.grid(True)

    if output_path is None:
        output_path = losses_json_path.parent / 'all_losses_plot.png'

    plt.savefig(output_path)
    plt.close()

    print(f"Loss plot saved to {output_path}")

    
def main():
    features = [
            'Area',
            # 'X',
            # 'Y',
            'Circularity',
            'Perimeter',
            'Displacement',
            'Skeleton Length', 
            'Skeleton Branch Points', 
            'Skeleton End Points', 
            'Skeleton Branch Length Mean', 
            'Skeleton Branch Length Std',
            'Skeleton Branch Length Max',
            'UMAP 1',
            'UMAP 2',
            # 'Mode 0',
            # 'Mode 1',
            # 'Mode 2',
            # 'Mode 3',
            # 'Mode 4',
            # 'Speed',
            'Phagocytes within 100 pixels',
            'Phagocytes within 250 pixels',
            'Phagocytes within 500 pixels',
            'Total Fluorescence', 
            'Fluorescence Distance Mean', 
            'Fluorescence Distance Variance',
            ] 
    
    model_params = {
        'input_size': len(features)*2,  # Features + mask
        'output_size': 8,  # Number of time bins
        'lstm_hidden_size': 32,
        'lstm_dropout': 0.0,
        'predictor_layers': [32],
        'attention_layers': [32, 32],
        'fc_layers': [32, 32],
        'features': features,
    }
    
    train(
        model=dynamic_deephit.DynamicDeepHit,
        model_params=model_params,
        train_hdf5_paths=[
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
            
        ],
        val_hdf5_paths=[
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
        ],
        features=features,
        optimiser=torch.optim.Adam,
        loss_fn=dynamic_deephit.compute_loss,
        num_epochs=50,
        save_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'dynamic_deephit' / 'test_run',
        batch_size=256,
        lr=1e-3
    )
    
    # train_single_dataset(
    #     model=dynamic_deephit.DynamicDeepHit,
    #     model_params=model_params,
    #     hdf5_path=Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
    #     features=features,
    #     optimiser=torch.optim.Adam,
    #     loss_fn=dynamic_deephit.compute_loss,
    #     num_epochs=10,
    #     save_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'dynamic_deephit' / 'test_run_13_06',
    #     batch_size=128,
    #     lr=1e-3,
    #     val_split=0.2,
    #     seed=42
    # )
    
if __name__ == '__main__':
    main()


