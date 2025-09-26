from pathlib import Path
import os
import json
from collections import defaultdict

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models import dynamic_deephit
from PhagoPred.survival_analysis.validate import validate_step, visualize_validation_predictions

def train_step(model, dataloader, optimiser, loss_fn, device):
    model.train()
    # losses = {'Total Loss': 0.0, 'NLL Loss': 0.0, 'Ranking Loss': 0.0, 'Prediction Loss': 0.0}
    # total_loss = 0.0
    losses = defaultdict(float)
    
    for batch in dataloader:
        cell_features, lengths, observation_times, event_indicators = batch
        cell_features = cell_features.to(device)
        lengths = lengths.to(device)
        observation_times = observation_times.to(device)
        event_indicators = event_indicators.to(device)

        optimiser.zero_grad()
        outputs, y = model(cell_features)
        loss_values = loss_fn(outputs, cell_features, y, observation_times, event_indicators)
        
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

    model = model(**model_params)
    
    with open(save_dir / 'model_params.json', 'w') as f:
        json.dump(model_params, f)
        
        
    train_dataset = CellDataset(
        hdf5_paths=train_hdf5_paths,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)

    validate_dataset = CellDataset(
        hdf5_paths=val_hdf5_paths,
        features=features,
    )

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    optimiser = optimiser(model.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)
    
    training_json  = save_dir / 'training.jsonl'
    with open(training_json, 'w') as f:
        # f.write('\t'.join(['Epoch', 'Train Loss', 'Validate Loss']) + '\n')
        for epoch in tqdm(range(1, num_epochs + 1), desc="Training"):
            train_losses = train_step(model, train_loader, optimiser, loss_fn, device)
            validate_losses = validate_step(model, validate_loader, loss_fn, device)
            losses_dict = {'epoch': epoch, 'train': train_losses, 'validate': validate_losses}
            f.write(json.dumps(losses_dict) + '\n')

    torch.save(model.state_dict(), save_dir / 'model.pth')

    plot_training_losses(training_json, save_dir / 'loss_plot.png')

    print(f"Training complete. Model saved to {save_dir}")
    

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
            'X',
            'Y',
            # 'Circularity',
            # 'Perimeter',
            # 'Displacement',
            # 'Mode 0',
            # 'Mode 1',
            # 'Mode 2',
            # 'Mode 3',
            # 'Mode 4',
            # 'Speed',
            # 'Phagocytes within 100 pixels',
            # 'Phagocytes within 250 pixels',
            # 'Phagocytes within 500 pixels',
            # 'Total Fluorescence', 
            # 'Fluorescence Distance Mean', 
            # 'Fluorescence Distance Variance',
            ] 
    
    model_params = {
        'input_size': len(features)*2,  # Features + mask
        'output_size': 501,  # Number of time bins
        'lstm_hidden_size': 64,
        'lstm_dropout': 0.0,
        'predictor_layers': [32],
        'attention_layers': [64, 64],
        'fc_layers': [64, 64],
        'features': features,
    }
    
    train(
        model=dynamic_deephit.DynamicDeepHit,
        model_params=model_params,
        train_hdf5_paths=[
            Path('PhagoPred') / 'Datasets' / '27_05_500.h5',
        ],
        val_hdf5_paths=[
            Path('PhagoPred') / 'Datasets' / '27_05_500.h5',
        ],
        features=features,
        optimiser=torch.optim.Adam,
        loss_fn=dynamic_deephit.compute_loss,
        num_epochs=5,
        save_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'dynamic_deephit' / 'test_run',
        batch_size=32,
        lr=1e-3
    )

    # loss_fn = torch.nn.CrossEntropyLoss()

