"""Frames around the manual and predicted death of wrongly predicted cells,
with the held out fold model's death probability over the track.

Usage (after any number of folds have finished):
    PYTHONPATH=. python temp/death_classifier/cv_small_onedeath_dwp025_smooth_dim03_253/plot_errors.py [num_cells]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PhagoPred.death_classifier.dataset import CellWindowDataset
from PhagoPred.death_classifier.model import load_model
from PhagoPred.death_classifier.predict import decode_outcome, outcome_probs

RUN = Path('temp/death_classifier/cv_small_onedeath_dwp025_smooth_dim03_253')
OUT = RUN / 'error_examples'
NUM_CELLS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OFFSETS = np.arange(-9, 10, 3)  # frames shown either side of each death
MANUAL, PREDICTED = '#0b0b0b', '#2a78d6'
TOLERANCE = 20

labels = json.loads((RUN / 'labels' / 'labels.json').read_text())
truth = {f"{f.split('.')[0]}_{c}": v
         for f, cs in labels.items() for c, v in cs.items()}
targeted = {f"{f.split('.')[0]}_{c}"
            for f, cs in json.loads(
                (RUN / 'labels' / 'labels_targeted.json').read_text()).items()
            for c in cs}

folds = []
for fold_dir in sorted(RUN.glob('fold_*')):
    path = fold_dir / 'last_val_predictions.csv'
    if path.exists():
        p = pd.read_csv(path)
        folds.append(p[~p['bad'].astype(bool)].assign(fold_dir=str(fold_dir)))
p = pd.concat(folds).set_index('cell')
first = {}
for cell in p.index:
    first[cell] = json.loads(
        Path(f'temp/death_crops_smooth/{cell}.json').read_text())['first_frame']
p['first'] = pd.Series(first)
p['t'] = [np.nan if truth[c] is None else max(truth[c], first[c])
          for c in p.index]


def kind(row):
    t, q = row['t'], row['predicted']
    if pd.isna(t) and pd.isna(q):
        return None
    if pd.isna(t):
        return 'alive predicted dead'
    if pd.isna(q):
        return 'dead predicted alive'
    if abs(q - t) <= TOLERANCE:
        return None
    if q == row['first'] and t > row['first'] + 5:
        return 'later death called dead at start'
    return 'wrong death frame'


p['kind'] = p.apply(kind, axis=1)
wrong = p[p['kind'].notna()].copy()
order = {'later death called dead at start': 0, 'dead predicted alive': 1,
         'alive predicted dead': 2, 'wrong death frame': 3}
wrong['order'] = wrong['kind'].map(order)
# Spread across error types: take cells round robin over the kinds
picked = (wrong.sort_values(['order', 'cell'])
          .groupby('kind', group_keys=False)[wrong.columns]
          .apply(lambda g: g.assign(rank=range(len(g))))
          .sort_values(['rank', 'order']).head(NUM_CELLS))
print(f'{len(wrong)} wrong of {len(p)} cells from {len(folds)} folds:')
print(wrong['kind'].value_counts().to_string())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT.mkdir(exist_ok=True)
plt.rcParams['font.family'] = 'serif'


def frames_row(axs, crops, first_frame, centre, colour, label, outline):
    """Phase crops at centre + OFFSETS (absolute frames), clipped to the
    track; the centre frame outlined if it's a death frame."""
    last = first_frame + len(crops) - 1
    for ax, offset in zip(axs, OFFSETS):
        frame = centre + offset
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(outline and offset == 0)
            spine.set_color(colour)
            spine.set_linewidth(2.5)
        if not first_frame <= frame <= last:
            continue  # before / after the track: left blank
        ax.imshow(crops[frame - first_frame, 0], cmap='gray', vmin=0,
                  vmax=255)
        ax.set_title(f'{frame}', fontsize=8,
                     fontweight='bold' if outline and offset == 0 else
                     'normal')
    # Label on the first axis, which stays (blank if before the track)
    axs[0].set_ylabel(label, fontsize=8)


for cell, row in picked.iterrows():
    model, cfg = load_model(Path(row['fold_dir']) / 'last.pt', device)
    dataset = CellWindowDataset(cfg.cache_dir, keys=[cell],
                                **cfg.dataset_kwargs())
    probs = outcome_probs(model, cfg, dataset, cell, device)
    num_frames = len(probs) - 1
    first_frame = int(row['first'])
    frames = first_frame + np.arange(num_frames)
    crops = np.asarray(dataset.crops[cell])
    last_frame = first_frame + num_frames - 1

    fig = plt.figure(figsize=(len(OFFSETS) * 1.5, 6.6))
    grid = fig.add_gridspec(3, len(OFFSETS), height_ratios=[1, 1, 1.25],
                            hspace=0.45, wspace=0.05)
    manual = row['t']  # labelled frame, or the first frame if before it
    pred = row['predicted']
    if pd.isna(manual):
        manual_label = 'Manual: alive\n(end of track)'
    elif truth[cell] < first_frame:
        manual_label = (f'Manual: {truth[cell]}\n(before track\n'
                        f'starts at {first_frame})')
    else:
        manual_label = f'Manual: {int(manual)}'
    # Alive (no death frame): show the end of the track instead
    frames_row([fig.add_subplot(grid[0, i]) for i in range(len(OFFSETS))],
               crops, first_frame,
               int(manual) if pd.notna(manual) else last_frame - OFFSETS[-1],
               MANUAL, manual_label, pd.notna(manual))
    frames_row([fig.add_subplot(grid[1, i]) for i in range(len(OFFSETS))],
               crops, first_frame,
               int(pred) if pd.notna(pred) else last_frame - OFFSETS[-1],
               PREDICTED, f'Predicted: {int(pred)}' if pd.notna(pred)
               else 'Predicted: alive\n(end of track)', pd.notna(pred))

    ax = fig.add_subplot(grid[2, :])
    ax.bar(frames, probs[:num_frames], width=1.0, color=PREDICTED,
           linewidth=0)
    if pd.notna(manual):
        ax.axvline(manual, color=MANUAL, ls='--', lw=1.2, label='Manual')
    if pd.notna(pred):
        ax.axvline(pred, color=PREDICTED, lw=1.2, label='Predicted')
    ax.set_xlim(first_frame - 1, last_frame + 1)
    ax.set_xlabel('Frame')
    ax.set_ylabel('P(death at frame)')
    ax.text(0.99, 0.95, f'P(survives) = {probs[num_frames]:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    if pd.notna(manual) or pd.notna(pred):
        ax.legend(fontsize=8, loc='upper left', frameon=False)
    assert decode_outcome(probs) == (num_frames if pd.isna(pred) else
                                     int(pred) - first_frame)

    fig.suptitle(f"{cell}: {row['kind']}"
                 f"{' (targeted)' if cell in targeted else ''}",
                 fontsize=11)
    fig.savefig(OUT / f'{cell}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('saved', OUT / f'{cell}.png', '|', row['kind'])
