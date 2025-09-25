from pathlib import Path

import torch

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models import dynamic_deephit

def train(model, dataloader, optimiser, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        cell_features, lengths, observation_times, event_indicators = batch
        cell_features = cell_features.to(device)
        lengths = lengths.to(device)
        observation_times = observation_times.to(device)
        event_indicators = event_indicators.to(device)

        optimiser.zero_grad()
        outputs, y = model(cell_features)
        loss = loss_fn(outputs, cell_features, y, observation_times, event_indicators)
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item() * cell_features.size(0)  # Multiply by batch size
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def validate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            cell_features, lengths, observation_times, event_indicators = batch
            cell_features = cell_features.to(device)
            lengths = lengths.to(device)
            observation_times = observation_times.to(device)
            event_indicators = event_indicators.to(device)

            outputs, y = model(cell_features)
            loss = loss_fn(outputs, cell_features, y, observation_times, event_indicators)
            total_loss += loss.item() * cell_features.size(0)  # Multiply by batch size
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CellDataset(
        hdf5_paths = [Path('PhagoPred') / 'Datasets' / '27_05_500.h5'],
        features = features,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    validate_dataset = CellDataset(
        hdf5_paths = [Path('PhagoPred') / 'Datasets' / '27_05_500.h5'],
        features = features,
    )
    
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
   
    model = dynamic_deephit.DynamicDeepHit(
        input_size=2*len(features),
        output_size=501,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = dynamic_deephit.compute_loss
    
    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_loader, optimiser, loss_fn, device)
        validate_loss = validate(model, validate_loader, loss_fn, device)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validate Loss: {validate_loss:.4f}')

    # loss_fn = torch.nn.CrossEntropyLoss()

