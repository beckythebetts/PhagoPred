"""Evaluate any checkpoint of a training run on that run's validation cells.

train.py only saves predictions for the epoch with the best val score, so
use this to look at other checkpoints (e.g. last.pt). Uses the run's saved
split.json so the val cells are the same as in training, and current labels
from the run's labels file (so any relabelling since training is used).

Usage:
    python -m PhagoPred.death_classifier.evaluate temp/death_classifier/run --checkpoint last.pt
"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PhagoPred.death_classifier.config import TrainConfig
from PhagoPred.death_classifier.dataset import CellWindowDataset, load_labels
from PhagoPred.death_classifier.model import DeathNet, load_model
from PhagoPred.death_classifier.plots import plot_bad_tracks, plot_death_frames
from PhagoPred.death_classifier.predict import predict_track
from PhagoPred.death_classifier.train import evaluate


def plot_traces(model: DeathNet, cfg: TrainConfig, dataset: CellWindowDataset,
                predictions: pd.DataFrame, device: torch.device,
                save_path: Path, ncols: int = 5) -> None:
    """p_dead along each good val cell's track, worst death frame first.

    Shows whether errors come from the model (probabilities switching at the
    wrong time, ramping slowly, flickering) or from decoding.
    """
    good = predictions[~predictions['bad'].astype(bool)].copy()
    # Sort worst first: wrong alive / dead state, then by frame error
    state_wrong = good['true'].isna() != good['predicted'].isna()
    good['sort'] = np.where(state_wrong, np.inf, good['error'].abs())
    good = good.sort_values('sort', ascending=False, na_position='last')

    nrows = int(np.ceil(len(good) / ncols))
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(nrows,
                            ncols,
                            figsize=(3.2 * ncols, 1.9 * nrows),
                            sharey=True,
                            squeeze=False)
    for ax, (_, row) in zip(axs.flat, good.iterrows()):
        key = row['cell']
        p_dead = predict_track(model, cfg, dataset, key, device)['p_dead']
        frames = dataset.meta[key]['first_frame'] + np.arange(len(p_dead))
        ax.plot(frames, p_dead, color='tab:blue', lw=1)
        if pd.notna(row['true']):
            ax.axvline(row['true'], color='black', ls='--', lw=1.2,
                       label='Manual')
        if pd.notna(row['predicted']):
            ax.axvline(row['predicted'], color='tab:red', lw=1.2,
                       label='Predicted')
        true = 'alive' if pd.isna(row['true']) else int(row['true'])
        pred = 'alive' if pd.isna(row['predicted']) else int(row['predicted'])
        ax.set_title(f'{key}: {true} → {pred}', fontsize=8,
                     color='black' if row['correct'] else 'tab:red')
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4)
    for ax in axs.flat[len(good):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('p_dead', fontsize=8)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=8)
    fig.suptitle('Per-frame p_dead, manual (dashed) vs predicted (red) death '
                 'frame, worst first', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.savefig(save_path, dpi=110)
    plt.close(fig)


def evaluate_checkpoint(run_dir: Path,
                        checkpoint: str = 'last.pt',
                        threshold: float = 0.5,
                        traces: bool = False) -> dict:
    """Evaluate a checkpoint, saving predictions and plots in run_dir.

    Outputs are named after the checkpoint, e.g. last_val_predictions.csv,
    last_death_frames.png, and last_traces.png if traces is True.

    Returns:
        Val metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg = load_model(run_dir / checkpoint, device)

    with open(run_dir / 'split.json', 'r') as f:
        split = json.load(f)
    val_ds = CellWindowDataset(cfg.cache_dir,
                               load_labels(cfg.labels_path),
                               split['val'],
                               bad_keys=split['bad'],
                               **cfg.dataset_kwargs())
    metrics, predictions = evaluate(model, cfg, val_ds, device)

    name = Path(checkpoint).stem
    predictions.to_csv(run_dir / f'{name}_val_predictions.csv', index=False)
    if cfg.bad_weight > 0:
        plot_bad_tracks(predictions, run_dir / f'{name}_bad_tracks.png',
                        threshold)
    if cfg.loss != 'none':
        plot_death_frames(predictions, run_dir / f'{name}_death_frames.png',
                          cfg.tolerance)
        if traces:
            plot_traces(model, cfg, val_ds, predictions, device,
                        run_dir / f'{name}_traces.png')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a checkpoint on its run\'s validation cells.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', default='last.pt')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='p_bad threshold for the confusion matrix.')
    parser.add_argument('--traces',
                        action='store_true',
                        help='Also plot p_dead along each val cell\'s track.')
    args = parser.parse_args()
    metrics = evaluate_checkpoint(args.run_dir, args.checkpoint,
                                  args.threshold, args.traces)
    print(json.dumps(metrics, indent=4))
