from pathlib import Path
import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import h5py

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models.temp_cnn import model
from PhagoPred.survival_analysis.models.temp_cnn.validate import validate_step, visualize_validation_predictions


def train_step(model, dataloader, optimiser, loss_fn, device, max_grad_norm=1.0):
    model.train()
    losses = defaultdict(float)

    for batch in dataloader:
        optimiser.zero_grad()
        outputs = model(batch['features'], batch['length'])
        loss_values = loss_fn(outputs, batch['time_to_event_bin'], batch['event_indicator'])

        loss = loss_values[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimiser.step()

        for key, value in zip(
            ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Censored Loss', 'Uncensored Loss'], loss_values
        ):
            losses[key] += value.item() * batch['features'].size(0)

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
        min_length=100,
        max_time_to_death=200,
        # uncensored_only=True,
    )

    bins = train_dataset.event_time_bins
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
        uncensored_only=False,
        min_length=100,
        max_time_to_death=200,
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, dataset=train_dataset, device=device), num_workers=0)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, dataset=validate_dataset, device=device), num_workers=0)
    
    compute_orcale_losses(validate_dataset, loss_fn, device)
    
    with open(save_dir / 'model_params.json', 'w') as f:
        json.dump(model_params, f)
        
    optimiser = optimiser(model.parameters(), lr=lr)
    training_json = save_dir / 'training.jsonl'
    
    training_json  = save_dir / 'training.jsonl'
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.9)
    
    with open(training_json, 'w') as f:
            for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
                train_losses = train_step(model, train_loader, optimiser, loss_fn, device)
                validate_losses = validate_step(model, validate_loader, loss_fn, device)
                
                losses_dict = {'epoch': epoch, 'train': train_losses, 'validate': validate_losses}
                print(losses_dict)
                f.write(json.dumps(losses_dict) + '\n')
                scheduler.step()

    # --- Step 7: Save model and plot ---
    torch.save(model.state_dict(), save_dir / 'model.pth')
    plot_training_losses(training_json, save_dir / 'loss_plot.png')
    print(f"Training complete. Model saved to {save_dir}")

    visualize_validation_predictions(model, validate_loader, device, num_examples=20, save_path=save_dir, bin_edges=bins, features=features, padding_at='start')


def compute_orcale_losses(dataset: CellDataset, loss_fn, device:str):
    """For given slice of each cell time series, compute loss using underlying pmf (if exists)."""
    losses = defaultdict(float)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=256,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, dataset=dataset, device=device),
        num_workers=0,
    )
    for batch in tqdm(dataloader, 'Getting oracle losses'):
        pmf = torch.tensor(batch['binned_pmf']).to(device)
        loss_values = loss_fn(
            None,
            batch['time_to_event_bin'],
            batch['event_indicator'],
            pmf=pmf,
            )
        for key, value in zip(
            ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Prediction Loss', 'Censored Loss', 'Uncensored Loss'], loss_values
        ):
            losses[key] += value.item() * pmf.size(0)
    loss_values = {key: value / len(dataloader.dataset)  for key, value in losses.items()}
    print(loss_values)
        
            
            
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

    def plot_normalised_loss(loss_type, colour):
        train = train_losses[loss_type]
        val = val_losses[loss_type]
        # maximum = max(train + val)
        maximum=1
        plt.plot(epochs, np.array(train)/maximum, label=f'Train {loss_type}', linestyle='-', color=colour)
        plt.plot(epochs, np.array(val)/maximum, label=f'Validation {loss_type}', linestyle='--', color=colour)
    cmap = plt.get_cmap('Set1')
    # Plot each loss type: train solid line, val dashed line
    loss_types = [_ for _ in loss_types if _ != 'Total Loss']
    plot_normalised_loss('Total Loss', 'k')
    for i, loss_type in enumerate(loss_types):
        colour = cmap(i)
        plot_normalised_loss(loss_type, colour)

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
    # features = [
    #         'Area',
    #         # 'X',
    #         # 'Y',
    #         'Circularity',
    #         'Perimeter',
    #         'Displacement',
    #         'Skeleton Length', 
    #         'Skeleton Branch Points', 
    #         'Skeleton End Points', 
    #         'Skeleton Branch Length Mean', 
    #         'Skeleton Branch Length Std',
    #         'Skeleton Branch Length Max',
    #         'Speed',
    #         'Alive Phagocytes within 100 pixels',
    #         'Alive Phagocytes within 250 pixels',
    #         'Alive Phagocytes within 500 pixels',
    #         'Dead Phagocytes within 100 pixels',
    #         'Dead Phagocytes within 250 pixels',
    #         'Dead Phagocytes within 500 pixels',
    #         # 'Total Fluorescence', 
    #         # 'Fluorescence Distance Mean', 
    #         # 'Fluorescence Distance Variance',
    #         ] 
    features = [
        '0',
        '1',
        '2',
        '3',
        ] 
    
    model_params = {
        'num_input_features': len(features),  # Features
        'output_size': 4,  # Number of time bins
        'num_channels': [64]*10,
        'kernel_sizes': [3]*10,
        'dilations': [2**n for n in range(10)],
        'attention_layers': [64, 64],
        'fc_layers': [64, 32],
        'features': features,
    }
    # datasets = list((Path('PhagoPred')/'Datasets'/ 'ExposureTest' / 'Truncated').iterdir())
    # train_datasets = datasets[:-2]
    # val_datasets = datasets[-2:]
    
    train_datasets = [Path('PhagoPred')/'Datasets'/'synthetic.h5']
    val_datasets = [Path('PhagoPred')/'Datasets'/'val_synthetic.h5']
    save_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'temp_cnn' / 'test_run'
    train(
        model=model.TemporalCNN,
        model_params=model_params,
        train_hdf5_paths=train_datasets,
        val_hdf5_paths=val_datasets,
        # train_hdf5_paths=[
        #     # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '21_10_2500.h5',
            
        # ],
        # val_hdf5_paths=[
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '21_10_2500.h5',
        #     # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
        #     # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
        # ],
        features=features,
        optimiser=torch.optim.Adam,
        loss_fn=model.compute_loss,
        num_epochs=100,
        save_dir=save_dir,
        batch_size=256,
        lr=1e-3
    )
    # compute_orcale_losses(val_datasets)
    # plot_training_losses(save_dir / 'training.jsonl', save_dir / 'loss_plot.png')
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


