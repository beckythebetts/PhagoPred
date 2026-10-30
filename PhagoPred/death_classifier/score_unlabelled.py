"""Rank unlabelled cells by how unsure the death model ensemble is, for
targeted labelling in label_gui.py.

Each cell is run through every fold's model from a cross validation run.
These cells aren't in any fold's training set, so all 5 models are honest
about them. The averaged outcome distribution (death in each 10 frame bin, or
survives) is scored by its entropy. That is high both when the models
disagree and when they agree the cell is ambiguous. Disagreement alone
(mutual information) and a few other summaries are saved too.

Crops are extracted in memory rather than to the crop cache, so scoring
doesn't need ~40 MB of disk per cell.

Usage:
    python -m PhagoPred.death_classifier.score_unlabelled \
        temp/death_labelling_02_10 \
        --run temp/death_classifier/cv_small_onedeath_dwp025_smooth_dim03 \
        --files 3.h5 17.h5 18.h5 19.h5
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from PhagoPred.death_classifier.config import TrainConfig
from PhagoPred.death_classifier.dataset import (CellWindowDataset,
                                                unlabelled_cells)
from PhagoPred.death_classifier.extract_crops import (cache_name,
                                                      extract_file)
from PhagoPred.death_classifier.model import DeathNet, load_ensemble
from PhagoPred.death_classifier.predict import decode_outcome, outcome_probs


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def uncertainty(probs: np.ndarray, first_frame: int,
                bin_size: int = 10) -> dict:
    """Summaries of an ensemble's outcome distributions for one cell.

    Frame probabilities are summed into bin_size frame bins first, so models
    a few frames apart on the death frame don't count as unsure.

    Args:
        probs: (num_models, T + 1) from outcome_probs.
        first_frame: the track's first frame, to give predictions as frames.
        bin_size: frames per bin.
    """
    num_frames = probs.shape[1] - 1
    binned = np.concatenate([
        np.add.reduceat(probs[:, :num_frames],
                        np.arange(0, num_frames, bin_size),
                        axis=1), probs[:, num_frames:]
    ],
                            axis=1)
    mean_binned = binned.mean(axis=0)
    entropy = float(_entropy(mean_binned))

    mean = probs.mean(axis=0)
    death_idx = decode_outcome(mean)
    model_idxs = np.array([decode_outcome(p) for p in probs])
    model_dies = model_idxs < num_frames
    return {
        'score': entropy,
        'disagreement': entropy - float(_entropy(binned).mean()),
        'p_dead': 1 - float(mean[num_frames]),
        'predicted':
        None if death_idx >= num_frames else first_frame + death_idx,
        'models_dead': int(model_dies.sum()),
        # Spread of the models' death frames, among those predicting death
        'frame_range':
        int(np.ptp(model_idxs[model_dies])) if model_dies.sum() > 1 else 0,
    }


def score_file(h5_path: Path,
               cells: list[int],
               ensemble: list[tuple[DeathNet, TrainConfig]],
               device: torch.device,
               cells_per_pass: int = 100) -> list[dict]:
    """Uncertainty of each cell in one h5 file, with crops made as for the
    ensemble's training."""
    cfg = ensemble[0][1]
    rows = []
    # A pass's crops are all held in memory, ~1 GB per 100 cells when
    # downsampled by 2
    for i in range(0, len(cells), cells_per_pass):
        tracks = extract_file(h5_path, cells[i:i + cells_per_pass],
                              **cfg.crop_kwargs())
        dataset = CellWindowDataset(tracks, **cfg.dataset_kwargs())
        for cell, track in tracks.items():
            key = cache_name(h5_path.name, cell)
            probs = np.stack([
                outcome_probs(model, model_cfg, dataset, key, device)
                for model, model_cfg in ensemble
            ])
            rows.append({
                'file': h5_path.name,
                'cell': cell,
                'first_frame': track['first'],
                'last_frame': track['last'],
                **uncertainty(probs, track['first']),
            })
    return rows


def priority_order(scores: pd.DataFrame,
                   start_frac: float = 0.5) -> pd.DataFrame:
    """Order cells to label: most uncertain, interleaved with cells predicted
    dead from the start of their track.

    Calling later deaths dead from the start is the most common error, and
    is often confident, so uncertainty alone misses those cells. Both queues
    are taken in decreasing score (least confident first).

    Args:
        scores: from score_file, with score, predicted and first_frame.
        start_frac: fraction of cells taken from the dead from start queue,
            while both have cells left.

    Returns:
        scores reordered, with a 'reason' column ('uncertain' or
        'dead_at_start').
    """
    scores = scores.sort_values('score', ascending=False)
    at_start = scores['predicted'] == scores['first_frame']
    queues = {
        'dead_at_start': list(scores.index[at_start]),
        'uncertain': list(scores.index[~at_start]),
    }
    order, reasons, taken = [], [], 0.0
    while queues['dead_at_start'] or queues['uncertain']:
        # Take from the dead from start queue whenever it is behind its share
        reason = ('dead_at_start' if queues['dead_at_start'] and
                  (not queues['uncertain'] or taken < start_frac *
                   (len(order) + 1)) else 'uncertain')
        taken += reason == 'dead_at_start'
        order.append(queues[reason].pop(0))
        reasons.append(reason)
    return scores.loc[order].assign(reason=reasons)


def score_unlabelled(dataset_dir: Path,
                     run_dir: Path,
                     files: list[str] | None = None,
                     labels_name: str = 'deaths.json',
                     out_name: str = 'label_priority.csv',
                     min_track_length: int = 50,
                     start_frac: float = 0.5) -> pd.DataFrame:
    """Score all unlabelled cells, saving them in labelling order (see
    priority_order).

    Args:
        dataset_dir: directory of h5 files and labels.
        run_dir: cross validation run whose fold models to use.
        files: h5 file names to score, defaults to all in dataset_dir.
        labels_name: labels file, as in label_gui.py.
        out_name: csv saved in dataset_dir, read by label_gui.py.
        min_track_length: as in label_gui.py.
        start_frac: share of cells predicted dead from the start, as in
            priority_order.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensemble = load_ensemble(run_dir, device=device)
    if any(cfg.loss != 'one_death' for _, cfg in ensemble):
        raise ValueError('Scoring needs one_death models, which give an '
                         'outcome distribution')
    labels_path = dataset_dir / labels_name
    labels, skipped = {}, {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
    skipped_path = labels_path.with_name(f'{labels_path.stem}_skipped.json')
    if skipped_path.exists():
        skipped = json.loads(skipped_path.read_text())

    if files is None:
        files = sorted(p.name for p in dataset_dir.glob('*.h5'))
    rows = []
    for name in tqdm(files, desc='Files'):
        with h5py.File(dataset_dir / name, 'r') as f:
            cells = unlabelled_cells(f, name, labels, skipped,
                                     min_track_length)
        rows += score_file(dataset_dir / name, cells, ensemble, device)
    scores = priority_order(pd.DataFrame(rows), start_frac)
    scores.to_csv(dataset_dir / out_name, index=False)
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Rank unlabelled cells by model uncertainty.')
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('--run',
                        type=Path,
                        help='Cross validation run directory (needed '
                        'unless --reorder).')
    parser.add_argument('--files',
                        nargs='+',
                        default=None,
                        help='h5 files to score (default all).')
    parser.add_argument('--labels', default='deaths.json')
    parser.add_argument('--out', default='label_priority.csv')
    parser.add_argument('--min-track-length', type=int, default=50)
    parser.add_argument('--start-frac',
                        type=float,
                        default=0.5,
                        help='Share of cells predicted dead from the start.')
    parser.add_argument('--reorder',
                        action='store_true',
                        help='Only reorder an existing --out file with '
                        '--start-frac, without rescoring.')
    args = parser.parse_args()
    if not args.reorder and args.run is None:
        parser.error('--run is needed to score cells')
    if args.reorder:
        path = args.dataset_dir.expanduser() / args.out
        scores = priority_order(pd.read_csv(path), args.start_frac)
        scores.to_csv(path, index=False)
    else:
        scores = score_unlabelled(args.dataset_dir.expanduser(), args.run,
                                  args.files, args.labels, args.out,
                                  args.min_track_length, args.start_frac)
    print(scores.head(20).to_string(index=False))
