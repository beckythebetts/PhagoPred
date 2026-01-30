import torch
from collections import defaultdict
from tqdm import tqdm


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    loss_config,
    device,
    max_grad_norm=1.0
):
    """
    Train for one epoch.

    Args:
        model: SurvivalModel instance
        dataloader: training data loader
        optimizer: optimizer instance
        loss_fn: compute_loss function from losses module
        loss_config: dict with loss weights
        device: torch device
        max_grad_norm: gradient clipping threshold

    Returns:
        dict with average losses
    """
    model.train()
    losses = defaultdict(float)
    num_samples = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Move batch to device if not already there
        features = batch['features'].to(device)
        lengths = batch['length'].to(device)
        t = batch['time_to_event_bin'].to(device)
        e = batch['event_indicator'].to(device)
        mask = batch.get('mask')
        if mask is not None:
            mask = mask.to(device)

        # Forward pass
        model_output = model(features, lengths, mask=mask)

        # Handle different return types (LSTM returns y_pred, CNN doesn't)
        if isinstance(model_output, tuple):
            outputs, y_pred = model_output[0], model_output[1] if len(model_output) > 1 else None
        else:
            outputs, y_pred = model_output, None

        # Compute PMF
        pmf = model.predict_pmf(outputs)

        # Compute loss
        loss_dict = loss_fn(
            pmf=pmf,
            t=t,
            e=e,
            y_pred=y_pred,
            y_true=features if y_pred is not None else None,
            mask=mask,
            loss_config=loss_config
        )

        # Backward pass
        loss = loss_dict['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        # Accumulate losses
        batch_size = features.size(0)
        num_samples += batch_size
        for key, value in loss_dict.items():
            losses[key] += value.item() * batch_size

    # Average losses
    avg_losses = {key: value / num_samples for key, value in losses.items()}
    return avg_losses


def validate_epoch(
    model,
    dataloader,
    loss_fn,
    loss_config,
    device
):
    """
    Validate for one epoch.

    Args:
        model: SurvivalModel instance
        dataloader: validation data loader
        loss_fn: compute_loss function
        loss_config: dict with loss weights
        device: torch device

    Returns:
        dict with average losses
    """
    model.eval()
    losses = defaultdict(float)
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device if not already there
            features = batch['features'].to(device)
            lengths = batch['length'].to(device)
            t = batch['time_to_event_bin'].to(device)
            e = batch['event_indicator'].to(device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)

            # Forward pass
            model_output = model(features, lengths, mask=mask)

            # Handle different return types
            if isinstance(model_output, tuple):
                outputs, y_pred = model_output[0], model_output[1] if len(model_output) > 1 else None
            else:
                outputs, y_pred = model_output, None

            # Compute PMF
            pmf = model.predict_pmf(outputs)

            # Compute loss
            loss_dict = loss_fn(
                pmf=pmf,
                t=t,
                e=e,
                y_pred=y_pred,
                y_true=features if y_pred is not None else None,
                mask=mask,
                loss_config=loss_config
            )

            # Accumulate losses
            batch_size = features.size(0)
            num_samples += batch_size
            for key, value in loss_dict.items():
                losses[key] += value.item() * batch_size

    # Average losses
    avg_losses = {key: value / num_samples for key, value in losses.items()}
    return avg_losses


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    loss_config,
    num_epochs,
    device,
    save_path=None,
    verbose=True
):
    """
    Full training loop with early stopping support.

    Args:
        model: SurvivalModel instance
        train_loader: training data loader
        val_loader: validation data loader
        optimizer: optimizer instance
        scheduler: learning rate scheduler (optional)
        loss_fn: compute_loss function
        loss_config: dict with loss weights
        num_epochs: number of epochs
        device: torch device
        save_path: path to save best model (optional)
        verbose: whether to print progress

    Returns:
        training_history: list of dicts with train/val losses per epoch
    """
    model = model.to(device)
    history = []
    best_val_loss = float('inf')

    iterator = tqdm(range(1, num_epochs + 1), desc="Training") if verbose else range(1, num_epochs + 1)

    for epoch in iterator:
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, loss_config, device)

        # Validate
        val_losses = validate_epoch(model, val_loader, loss_fn, loss_config, device)

        # Record history
        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses
        })

        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'normalization_means': train_loader.dataset.means,
            'normalization_stds': train_loader.dataset.stds,
        }, save_path)


        # Print progress
        if verbose:
            iterator.set_postfix({
                'train_loss': f"{train_losses['total']:.4f}",
                'val_loss': f"{val_losses['total']:.4f}"
            })

    return history
