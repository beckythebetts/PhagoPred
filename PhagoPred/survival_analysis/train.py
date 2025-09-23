from pathlib import Path

import torch

from PhagoPred.survival_analysis.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_analysis.models.dynamic_deephit import DynamicDeepHit

def train(model, dataloader, optimiser, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimiser.zero_grad()
        outputs, _ = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimiser.step()
        
        total_loss += loss.item() * inputs.size(1)  # Multiply by batch size
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def main():

    features = [
            'Area',
            'Circularity',
            'Perimeter',
            'Displacement',
            'Mode 0',
            'Mode 1',
            'Mode 2',
            'Mode 3',
            'Mode 4',
            'Speed',
            'Phagocytes within 100 pixels',
            'Phagocytes within 250 pixels',
            'Phagocytes within 500 pixels',
            'Total Fluorescence', 
            'Fluorescence Distance Mean', 
            'Fluorescence Distance Variance',
            ] 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = CellDataset(
        hdf5_paths = [Path('PhagoPred') / 'Datasets' / '27_05.h5'],
        features = features,
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x))

    model = DynamicDeepHit(
        input_size=2*len(features),
        output_size=10,
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    # loss_fn = torch.nn.CrossEntropyLoss()   

