"""Train DeathNet on cached cell crops.

Trains on random windows of frames (see dataset.py) and evaluates on whole
tracks of held out cells, decoding a single death frame per cell.

Usage:
    python -m PhagoPred.death_classifier.train \
        --labels ~/thor_server/MacrophageData/02_10/deaths.json
"""
from dataclasses import asdict
from pathlib import Path
from typing import Optional
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from PhagoPred.death_classifier.config import TrainConfig, num_input_channels
from PhagoPred.death_classifier.dataset import (CellWindowDataset,
                                                kfold_split, load_bad_tracks,
                                                load_labels, split_cells,
                                                worker_init_fn)
from PhagoPred.death_classifier.plots import plot_run
from PhagoPred.death_classifier.model import (DeathNet, frame_loss,
                                              one_death_loss)
from PhagoPred.death_classifier.predict import predict_track


def _subsample_cells(keys: list[str], labels: dict, bad_keys: list[str],
                     frac: float, seed: int) -> list[str]:
    """Stratified (dead / alive / bad) subset of cells, nested across fracs.

    Each group is shuffled in a fixed order (independent of frac) and the
    first ceil(frac * n) taken, so e.g. the 25% subset is inside the 50%.
    """
    rng = np.random.default_rng(seed + 1000)
    bad = set(bad_keys)
    groups = [
        [k for k in keys if k not in bad and labels.get(k) is not None],
        [k for k in keys if k not in bad and labels.get(k) is None],
        [k for k in keys if k in bad],
    ]
    subset = []
    for group in groups:
        group = sorted(group)
        rng.shuffle(group)
        subset += group[:int(np.ceil(frac * len(group)))]
    return subset


def _compute_loss(cfg: TrainConfig,
                  frame_logits,
                  survive_logit,
                  bad_logit,
                  batch,
                  dead_before_logit=None):
    """Bad track loss on all samples, death losses only on good tracks."""
    loss = torch.zeros((), device=bad_logit.device)
    if cfg.bad_weight > 0:
        loss = loss + cfg.bad_weight * F.binary_cross_entropy_with_logits(
            bad_logit, batch['bad'])
    good = batch['has_death_label']
    if cfg.loss != 'none' and good.any():
        valid = batch['valid'][good]
        if cfg.loss in ('frame', 'both'):
            loss = loss + frame_loss(frame_logits[good], batch['dead'][good],
                                     valid)
        if cfg.loss in ('one_death', 'both'):
            loss = loss + one_death_loss(
                frame_logits[good], survive_logit[good],
                batch['death_idx'][good], valid, cfg.sigma,
                None if dead_before_logit is None else dead_before_logit[good],
                batch['dead_before'][good])
    return loss


def _bce(p: np.ndarray, target: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))


def _auc(p: np.ndarray, target: np.ndarray) -> float:
    return float(roc_auc_score(target,
                               p)) if 0 < target.mean() < 1 else float('nan')


def evaluate(model: DeathNet, cfg: TrainConfig, dataset: CellWindowDataset,
             device: torch.device) -> tuple[dict, pd.DataFrame]:
    """Whole track metrics on held out cells.

    Death metrics are over good tracks only, bad track metrics over all.
    """
    model.eval()
    rows, all_p, all_dead = [], [], []
    for key in dataset.keys:
        result = predict_track(model, cfg, dataset, key, device)
        bad = key in dataset.bad_keys
        true = None if bad else dataset.labels[key]
        if true is not None:
            # Cells labelled dead before their track starts (e.g. frame 0
            # for a track starting at 160) can at best be predicted dead
            # from the first tracked frame, as in training
            true = max(true, dataset.meta[key]['first_frame'])
        predicted = result['predicted']
        if bad:
            correct, error = None, None
        elif true is None or predicted is None:
            correct = true is None and predicted is None
            error = None
        else:
            error = predicted - true
            correct = abs(error) <= cfg.tolerance
        rows.append({
            'cell': key,
            'bad': bad,
            'p_bad': round(result['p_bad'], 3),
            'true': true,
            'predicted': predicted,
            'error': error,
            'correct': correct,
        })
        if not bad:
            all_p.append(result['p_dead'])
            all_dead.append(result['dead'])

    rows = pd.DataFrame(rows)
    p, dead = np.concatenate(all_p), np.concatenate(all_dead)
    p_bad, is_bad = rows['p_bad'].to_numpy(), rows['bad'].to_numpy(float)
    metrics = {
        'val_bce': _bce(p, dead),
        'val_auc': _auc(p, dead),
        'val_cells_correct': int(rows['correct'].eq(True).sum()),
        'val_good_cells': int((~rows['bad']).sum()),
        'val_bad_bce': _bce(p_bad, is_bad),
        'val_bad_auc': _auc(p_bad, is_bad),
    }
    # Select only on the losses being trained
    metrics['val_score'] = cfg.bad_weight * metrics['val_bad_bce']
    if cfg.loss != 'none':
        metrics['val_score'] += metrics['val_bce']
    return metrics, rows


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg.labels_path)
    bad_keys = load_bad_tracks(cfg.labels_path)
    # Skip labelled cells which haven't been extracted yet, so labelling can
    # carry on while training
    uncached = [
        k for k in set(labels) | set(bad_keys)
        if not (cfg.cache_dir / f'{k}.npy').exists()
    ]
    if uncached:
        print(f'Skipping {len(uncached)} labelled cells not in cache '
              f'(run extract_crops.py): {sorted(uncached)[:5]}')
        labels = {k: v for k, v in labels.items() if k not in uncached}
        bad_keys = [k for k in bad_keys if k not in uncached]
    if cfg.loss == 'none' and cfg.bad_weight <= 0:
        raise ValueError('Nothing to train: loss is none and bad_weight is 0')
    if cfg.bad_weight <= 0:
        # Death only model, to run after a bad track model has filtered out
        # bad tracks, so train and evaluate on good tracks only
        print(f'bad_weight is 0, leaving out {len(bad_keys)} bad tracks')
        labels = {k: v for k, v in labels.items() if k not in set(bad_keys)}
        bad_keys = []
    if cfg.n_folds > 0:
        train_keys, val_keys = kfold_split(labels, bad_keys, cfg.n_folds,
                                           cfg.fold, cfg.seed)
    else:
        train_keys, val_keys = split_cells(labels, bad_keys, cfg.val_frac,
                                           cfg.seed)
    if cfg.train_frac < 1:
        train_keys = _subsample_cells(train_keys, labels, bad_keys,
                                      cfg.train_frac, cfg.seed)
    num_val_bad = len(set(val_keys) & set(bad_keys))
    print(f'{len(train_keys)} train cells, {len(val_keys)} val cells '
          f'({num_val_bad} bad tracks)')
    # Save the split so evaluate.py uses the same val cells even after more
    # cells are labelled
    with open(cfg.out_dir / 'split.json', 'w') as f:
        json.dump({'train': train_keys, 'val': val_keys, 'bad': bad_keys}, f)

    train_ds = CellWindowDataset(cfg.cache_dir,
                                 labels,
                                 train_keys,
                                 bad_keys=bad_keys,
                                 **cfg.dataset_kwargs(training=True))
    # No val cells (val_frac 0, n_folds 0) for a final model trained on all
    # labels, with its epochs fixed in advance: last.pt is the model
    val_ds = CellWindowDataset(cfg.cache_dir,
                               labels,
                               val_keys,
                               bad_keys=bad_keys,
                               **cfg.dataset_kwargs()) if val_keys else None
    # Crop settings, from the cache json (lists from json as tuples)
    settings = {
        tuple((name, tuple(meta[name]) if isinstance(meta[name], list) else
               meta[name]) for name in TrainConfig._FROM_CACHE)
        for ds in (train_ds, val_ds) if ds is not None
        for meta in ds.meta.values()
    }
    if len(settings) > 1:
        raise ValueError(f'Cells in {cfg.cache_dir} were extracted with '
                         f'different settings: {settings}')
    for name, value in settings.pop():
        setattr(cfg, name, value)
    cfg.to_json(cfg.out_dir / 'config.json')

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None)

    model = DeathNet(in_channels=num_input_channels(cfg),
                     widths=cfg.widths,
                     hidden=cfg.hidden,
                     dropout=cfg.dropout,
                     bad_pool_r=cfg.bad_pool_r,
                     frame_norm=cfg.frame_norm,
                     dead_before=cfg.dead_before).to(device)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters')
    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    history = []
    best_score = float('inf')
    for epoch in range(cfg.epochs):
        model.train()
        start = time.time()
        train_losses = []
        for batch in train_loader:
            batch = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            with torch.autocast(device.type, enabled=cfg.amp):
                frame_logits, survive_logit, bad_logit, _, dead_before_logit = (
                    model(batch['x'].float(), batch['valid']))
            loss = _compute_loss(
                cfg, frame_logits.float(), survive_logit.float(),
                bad_logit.float(), batch, None
                if dead_before_logit is None else dead_before_logit.float())
            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            train_losses.append(loss.item())
        scheduler.step()

        checkpoint = {'model': model.state_dict(), 'config': asdict(cfg)}
        torch.save(checkpoint, cfg.out_dir / 'last.pt')
        evaluate_now = val_ds is not None and (
            (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1)
        metrics = {
            'epoch': epoch,
            'train_loss': float(np.mean(train_losses)),
        }
        if evaluate_now:
            val_metrics, predictions = evaluate(model, cfg, val_ds, device)
            metrics.update(val_metrics)
        metrics['time'] = time.time() - start
        history.append(metrics)
        pd.DataFrame(history).to_csv(cfg.out_dir / 'history.csv', index=False)
        if not evaluate_now:
            print(f"Epoch {epoch:3d} | train loss {metrics['train_loss']:.4f} "
                  f"| {metrics['time']:.0f}s")
            continue
        print(f"Epoch {epoch:3d} | train loss {metrics['train_loss']:.4f} | "
              f"val bce {metrics['val_bce']:.4f} | val auc "
              f"{metrics['val_auc']:.3f} | val cells correct "
              f"{metrics['val_cells_correct']}/{metrics['val_good_cells']} | "
              f"val bad bce {metrics['val_bad_bce']:.4f} | val bad auc "
              f"{metrics['val_bad_auc']:.3f} | {metrics['time']:.0f}s")

        # Select on death and bad track losses combined
        if metrics['val_score'] < best_score:
            best_score = metrics['val_score']
            torch.save(checkpoint, cfg.out_dir / 'best.pt')
            predictions.to_csv(cfg.out_dir / 'best_val_predictions.csv',
                               index=False)

    if val_ds is None:
        print(f'\nNo val cells, final model is {cfg.out_dir / "last.pt"}')
        return
    print(f'\nBest val score {best_score:.4f}, predictions:')
    print(
        pd.read_csv(cfg.out_dir /
                    'best_val_predictions.csv').to_string(index=False))
    plot_run(cfg.out_dir)


def _parse_args(argv: Optional[list[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description='Train DeathNet.')
    defaults = TrainConfig(labels_path=Path())
    for name, value in asdict(defaults).items():
        flag = f"--{name.replace('_', '-')}"
        if name in TrainConfig._FROM_CACHE:
            continue  # set from the cache by train()
        if name == 'labels_path':
            parser.add_argument('--labels',
                                dest=name,
                                type=Path,
                                required=True)
        elif isinstance(value, bool):
            parser.add_argument(flag,
                                dest=name,
                                action=argparse.BooleanOptionalAction,
                                default=value)
        elif isinstance(value, tuple):
            parser.add_argument(flag,
                                dest=name,
                                nargs='+',
                                type=type(value[0]),
                                default=value)
        else:
            parser.add_argument(flag,
                                dest=name,
                                type=type(value),
                                default=value)
    args = vars(parser.parse_args(argv))
    args['labels_path'] = args['labels_path'].expanduser()
    for name, value in args.items():
        if isinstance(getattr(defaults, name), tuple):
            args[name] = tuple(value)
    return TrainConfig(**args)


if __name__ == '__main__':
    train(_parse_args())
