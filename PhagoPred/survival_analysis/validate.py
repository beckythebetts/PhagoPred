from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
import json

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models import dynamic_deephit

def validate_step(model, dataloader, loss_fn, device):
    model.eval()
    # total_loss = 0.0
    losses = defaultdict(float)

    with torch.no_grad():
        for batch in dataloader:
            cell_features, lengths, observation_times, event_indicators = batch
            cell_features = cell_features.to(device)
            lengths = lengths.to(device)
            observation_times = observation_times.to(device)
            event_indicators = event_indicators.to(device)

            outputs, y = model(cell_features)
            loss_values = loss_fn(outputs, cell_features, y, observation_times, event_indicators)
            for key, value in zip(
                ['Total Loss', 'NLL Loss', 'Ranking Loss', 'Prediction Loss'], loss_values):
                losses[key] += value.item() * cell_features.size(0)  # Multiply by batch size

    avg_losses = {key: value / len(dataloader.dataset) for key, value in losses.items()}
    return avg_losses

def visualize_validation_predictions(model, dataloader, device, num_examples=5, save_path=None):
    model.eval()
    model.to(device)

    examples_plotted = 0

    with torch.no_grad():
        for batch in dataloader:
            cell_features, lengths, observation_times, event_indicators = batch
            cell_features = cell_features.to(device)
            observation_times = observation_times.cpu().numpy()
            event_indicators = event_indicators.cpu().numpy()

            # Get predicted distribution over time bins
            predicted_dists, _ = model(cell_features)  # Shape: (batch_size, time_bins)

            predicted_dists = predicted_dists.cpu().numpy()

            for i in range(len(cell_features)):
                if examples_plotted >= num_examples:
                    break

                pred_dist = predicted_dists[i]
                true_time = observation_times[i]
                event = event_indicators[i]

                plt.figure(figsize=(10, 4))
                plt.plot(pred_dist, label='Predicted Distribution')
                if event == 1:
                    plt.axvline(x=true_time, color='red', linestyle='--', label=f'True Death @ {true_time}')
                else:
                    plt.axvline(x=true_time, color='orange', linestyle='--', label=f'Censored @ {true_time}')

                plt.xlabel('Time Frame')
                plt.ylabel('Probability')
                plt.title(f'Cell {i+1} - Predicted Survival Distribution')
                plt.legend()
                plt.grid(True)

                if save_path:
                    path = save_path / f'val_pred_cell_{examples_plotted+1}.png'
                    plt.savefig(path)
                    print(f"Saved: {path}")
                    plt.close()
                else:
                    plt.show()

                examples_plotted += 1

            if examples_plotted >= num_examples:
                break


def validate(model, model_dir, val_hdf5_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(model_dir / 'model_params.json', 'r') as f:
        model_params = json.load(f)

    features = model_params['features']
    model = model(**model_params)
    model.load_state_dict(torch.load(model_dir / 'model.pth', map_location=device))
    model.to(device)

    validate_dataset = CellDataset(
        hdf5_paths=val_hdf5_paths,
        features=features
    )

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    loss_fn = dynamic_deephit.compute_loss

    val_loss = validate_step(model, validate_loader, loss_fn, device)
    print(f"Validation Loss: {val_loss:.4f}")

    visualize_validation_predictions(model, validate_loader, device, num_examples=5, save_path=model_dir)
    
def main():
    validate(
        model=dynamic_deephit.DynamicDeepHit,
        model_dir=Path('PhagoPred') / 'survival_analysis' / 'models' / 'dynamic_deephit' / 'test_run',
        val_hdf5_paths=[
            Path('PhagoPred') / 'Datasets' / '27_05_500.h5',
        ]
    )