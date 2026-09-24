from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from copy import deepcopy

import torch

from tqdm import tqdm

from PhagoPred.survival_v2.models import SurvivalModel
from PhagoPred.survival_v2.losses import compute_loss
from PhagoPred.survival_v2.data.dataset import CellSample
from PhagoPred.survival_v2.data.survival_dataset import SurvivalCellBatch
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
    num_events = 0
    num_batches = 0
    min_events_per_batch = None
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

        # Track event/censoring balance (event=1, censored=0). Survival batches
        # expose `event_indicator`; binary batches expose `event`.
        ev = getattr(batch, 'event_indicator', None)
        if ev is None:
            ev = getattr(batch, 'event', None)
        if ev is not None:
            batch_events = int((ev == 1).sum().item())
            num_events += batch_events
            num_batches += 1
            min_events_per_batch = (batch_events
                                    if min_events_per_batch is None else min(
                                        min_events_per_batch, batch_events))

        for key, val in batch_losses.items():
            losses[key] += val.item() * batch_size

    # average losses, plus event-balance diagnostics
    metrics = {key: value / num_samples for key, value in losses.items()}
    metrics['event_fraction'] = num_events / max(num_samples, 1)
    metrics['avg_events_per_batch'] = num_events / max(num_batches, 1)
    metrics['min_events_per_batch'] = (min_events_per_batch if
                                       min_events_per_batch is not None else 0)
    return metrics


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
) -> tuple[dict, dict]:
    """Train a pytrch binary/survival model"""
    model = model.to(device)
    history = []

    best_val = None
    best_model = deepcopy(model.state_dict())

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
            val = validate_losses['total']
            if best_val is None:
                best_val = val
            if val < best_val:
                best_val = val
                best_model = deepcopy(model.state_dict())

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
                f"{validate_losses.get('total', float('nan')):.4f}",
                'ev_frac':
                f"{train_losses.get('event_fraction', float('nan')):.2f}",
                'min_ev/batch':
                int(train_losses.get('min_events_per_batch', 0)),
            })

    # Restore the best-validation weights into the in-place model so the
    # caller's `model` object matches `best_model` (and the saved state dict),
    # even when it is reused for evaluation without reloading from disk.
    model.load_state_dict(best_model)

    return history, best_model
