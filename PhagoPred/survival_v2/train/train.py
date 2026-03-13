from __future__ import annotations
from pathlib import Path
import json

from torch.utils.data import DataLoader
import torch
from PhagoPred.survival_v2.configs import TrainingCfg, LossCfg
from PhagoPred.survival_v2.models import SurvivalModel, ClassicalSurvivalModel
from PhagoPred.survival_v2.data import CellDataset, binary_collate_fn, survival_collate_fn, BinaryCellDataset
from PhagoPred.survival_v2.utils.plots import plot_losses
from PhagoPred.utils.tools import to_json_safe
from .train_classical import train_classical
from .train_deep import train_deep


def train(
    model: SurvivalModel | ClassicalSurvivalModel,
    train_dataset: CellDataset,
    val_dataset: CellDataset,
    save_dir: Path,
    train_config: TrainingCfg = None,
    loss_cfg: LossCfg = None,
    device: str = 'cpu',
) -> dict:
    """Train either a survival/binary deep/classical model."""
    if isinstance(model, SurvivalModel):
        if isinstance(train_dataset, BinaryCellDataset):
            collate_fn = binary_collate_fn
        else:
            collate_fn = survival_collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, device=device))
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, device=device))

        optimiser = torch.optim.Adam(model.parameters())
        scheduler_type = train_config.scheduler
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimiser,
                step_size=train_config.step_size,
                gamma=train_config.gamma,
            )
        else:
            scheduler = None

        history = train_deep(
            model,
            train_loader,
            val_loader,
            optimiser,
            scheduler,
            loss_cfg,
            train_config.num_epochs,
            device=device,
        )

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'normalization_means': train_loader.dataset.means,
                'normalization_stds': train_loader.dataset.stds,
            }, save_dir / 'model.pkl')

        with (save_dir / 'training_history.json').open('w') as f:
            json.dump(history, f)

        plot_losses(history, save_dir / 'training_losses.png')

    elif isinstance(model, ClassicalSurvivalModel):
        history = train_classical(model, train_dataset)

        with (save_dir / 'training_history.json').open('w') as f:
            json.dump(history, f)

        model.save(str(save_dir / 'model.pkl'))

    else:
        raise TypeError(
            'Model must be Survival Model or ClassicalSurvivalModel subclass')

    return history
