from typing import Optional, Union

from pathlib import Path
import torch

from PhagoPred.prediction.datasets import CellDataset, SummaryStatsCellDataset

def get_data_loaders(dataset: Union[CellDataset, SummaryStatsCellDataset],
                     batch_size: int = 32,
                     shuffle: bool = True,
                     train_ratio: float = 0.8,
):
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator()
    train_datset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_datset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader