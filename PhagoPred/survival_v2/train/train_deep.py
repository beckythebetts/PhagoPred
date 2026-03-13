from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import torch

from tqdm import tqdm

from PhagoPred.survival_v2.models import SurvivalModel
from PhagoPred.survival_v2.losses import compute_loss
from PhagoPred.survival_v2.data.dataset import CellSample
from PhagoPred.survival_v2.configs.losses import LossCfg
from PhagoPred.utils.logger import get_logger

log = get_logger()


def epoch(
    model: SurvivalModel,
    dataloader: torch.utils.data.DataLoader,
    loss_cfg: LossCfg,
    optimiser: torch.optim.Optimizer | None = None,
    max_grad_norm: float = 1.0,
    training: bool = True,
) -> dict:
    """
    Train the model for one epoch.

    Args:
        model: SurvivalModel instance
        dataloader: DataLoader for training data
        optimiser: optimizer instance
        loss_func: function to compute loss
        loss_cfg: dict with loss configuration (e.g. weights)
        device: torch device to use for training
        max_grad_norm: maximum norm for gradient clipping
    Returns:
        dict with average losses for the epoch
    """
    if training:
        model.train()
    else:
        model.eval()
    num_samples = 0
    losses = defaultdict(float)

    for batch in dataloader:
        batch: CellSample
        optimiser.zero_grad()

        model_output = model(batch.features, batch.length, mask=batch.mask)

        # Handle different return types (LSTM returns y_pred, CNN doesn't)
        if isinstance(model_output, tuple):
            outputs, y_pred = model_output[0], model_output[1] if len(
                model_output) > 1 else None
        else:
            outputs, y_pred = model_output, None

        batch_losses = compute_loss(outputs, batch, loss_cfg, y_pred)

        if training:
            batch_loss = batch_losses['total']
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_grad_norm)
            optimiser.step()

        batch_size = batch.features.size(0)
        num_samples += batch_size
        for key, val in batch_losses.items():
            losses[key] += val.item() * batch_size

    # average losses
    return {key: value / num_samples for key, value in losses.items()}


def train_deep(
    model: SurvivalModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_cfg: LossCfg,
    num_epochs: int,
    device: str,
    verbose: bool = True,
    validate_every: int = 1,
) -> dict:
    """Train a pytrch binary/survival model"""
    model = model.to(device)
    history = []
    progress_bar = tqdm(range(1, num_epochs +
                              1), desc="Training") if verbose else range(
                                  1, num_epochs + 1)

    for epoch_idx in progress_bar:
        train_losses = epoch(model, train_loader, loss_cfg, optimiser)
        validate_losses = {}
        if epoch_idx % validate_every == 0:
            validate_losses = epoch(model,
                                    val_loader,
                                    loss_cfg,
                                    optimiser,
                                    training=False)

        log.info(
            f'Epoch {epoch_idx}\n\ttrain losses: {train_losses}\n\tvalidate losses: {validate_losses}'
        )

        history.append({
            'epoch': epoch_idx,
            'train': train_losses,
            'val': validate_losses
        })

        if scheduler is not None:
            scheduler.step()

        if verbose:
            progress_bar.set_postfix({
                'train_loss':
                f"{train_losses['total']:.4f}",
                'val_loss':
                f"{validate_losses.get('total', float('nan')):.4f}"
            })

    return history
