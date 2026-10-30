"""Plots of a training run's validation predictions.

Reads best_val_predictions.csv from a run directory (saved by train.py at
the best epoch), so works for any finished run without reloading the model.

Usage:
    python -m PhagoPred.death_classifier.plots temp/death_classifier/bad_model
"""
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_bad_tracks(predictions: pd.DataFrame,
                    save_path: Path,
                    threshold: float = 0.5) -> None:
    """Confusion matrix of bad track predictions, and p_bad per class.

    Args:
        predictions: per cell predictions with 'bad' and 'p_bad' columns.
        save_path: where to save the figure.
        threshold: p_bad above which a track is predicted bad.
    """
    is_bad = predictions['bad'].astype(bool).to_numpy()
    p_bad = predictions['p_bad'].to_numpy()
    matrix = confusion_matrix(is_bad, p_bad > threshold, labels=[False, True])

    plt.rcParams["font.family"] = 'serif'
    fig, (ax_cm, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

    ax_cm.imshow(matrix, cmap='Blues')
    for (i, j), count in np.ndenumerate(matrix):
        row_frac = count / max(matrix[i].sum(), 1)
        ax_cm.text(j, i, f'{count}\n({row_frac:.0%})', ha='center',
                   va='center',
                   color='white' if count > matrix.max() / 2 else 'black')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_xticklabels(['Good', 'Bad'])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_yticklabels(['Good', 'Bad'])
    ax_cm.set_xlabel(f'Predicted (p_bad > {threshold})')
    ax_cm.set_ylabel('Manual label')
    ax_cm.set_title('Bad track confusion matrix')

    bins = np.linspace(0, 1, 21)
    ax_hist.hist(p_bad[~is_bad], bins=bins, alpha=0.6, color='tab:grey',
                 label=f'Good ({(~is_bad).sum()})')
    ax_hist.hist(p_bad[is_bad], bins=bins, alpha=0.6, color='tab:red',
                 label=f'Bad ({is_bad.sum()})')
    ax_hist.axvline(threshold, color='black', ls='--', lw=1)
    ax_hist.set_xlabel('p_bad')
    ax_hist.set_ylabel('Val cells')
    ax_hist.set_title('Predicted bad track probability')
    ax_hist.legend()

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_death_frames(predictions: pd.DataFrame,
                      save_path: Path,
                      tolerance: int = 5,
                      num_frames: int = 288,
                      marker_size: float = 18) -> None:
    """True vs predicted death frame for good tracks.

    Cells which stay alive (manual or predicted) are plotted in an 'Alive'
    column / row beyond the last frame.

    Args:
        predictions: per cell predictions with 'bad', 'true', 'predicted'.
        save_path: where to save the figure.
        tolerance: frames either side of the diagonal counted as correct
            (alive cells are correct if predicted alive).
        num_frames: frames per movie, used to position the 'Alive' row.
        marker_size: scatter marker size.
    """
    good = predictions[~predictions['bad'].astype(bool)]
    alive_pos = num_frames + 30
    true = good['true'].fillna(alive_pos).to_numpy()
    predicted = good['predicted'].fillna(alive_pos).to_numpy()
    both_die = (true < alive_pos) & (predicted < alive_pos)
    # Recomputed for this tolerance rather than using the saved 'correct'
    # column, which is for the tolerance used in training
    correct = ((good['true'].isna() & good['predicted'].isna()) |
               ((good['predicted'] - good['true']).abs() <= tolerance)
               ).to_numpy()

    # Small jitter so cells with the same alive / dead state don't overlap
    rng = np.random.default_rng(0)
    jitter = lambda v: v + np.where(v == alive_pos,
                                    rng.uniform(-8, 8, len(v)), 0)

    plt.rcParams["font.family"] = 'serif'
    fig, ax = plt.subplots(figsize=(6.5, 6))
    frames = np.array([0, num_frames])
    ax.fill_between(frames, frames - tolerance, frames + tolerance,
                    color='tab:green', alpha=0.15,
                    label=f'±{tolerance} frames')
    ax.plot(frames, frames, color='black', lw=0.8)
    ax.scatter(jitter(true[correct]), jitter(predicted[correct]),
               s=marker_size,
               color='tab:green', label=f'Correct ({correct.sum()})')
    ax.scatter(jitter(true[~correct]), jitter(predicted[~correct]),
               s=marker_size,
               color='tab:red', label=f'Wrong ({(~correct).sum()})')
    # Separate the 'Alive' row / column from the frame axis
    ax.axvline(alive_pos - 15, color='grey', lw=0.5)
    ax.axhline(alive_pos - 15, color='grey', lw=0.5)

    ticks = list(range(0, num_frames + 1, 50))
    for set_ticks, set_labels in ((ax.set_xticks, ax.set_xticklabels),
                                  (ax.set_yticks, ax.set_yticklabels)):
        set_ticks(ticks + [alive_pos])
        set_labels([str(t) for t in ticks] + ['Alive'])
    ax.set_xlim(-10, alive_pos + 15)
    ax.set_ylim(-10, alive_pos + 15)
    ax.set_xlabel('Manual death frame')
    ax.set_ylabel('Predicted death frame')

    errors = np.abs(predicted[both_die] - true[both_die])
    state_correct = ((true == alive_pos) == (predicted == alive_pos)).mean()
    title = (f'{correct.sum()}/{len(good)} within ±{tolerance} frames, '
             f'alive/dead state {state_correct:.0%} correct')
    if len(errors):
        title += (f'\nmedian error where both die: {np.median(errors):.0f} '
                  f'frames')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper left', fontsize='small')
    ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_history(history: pd.DataFrame,
                 save_path: Path,
                 death: bool = True,
                 bad: bool = True) -> None:
    """Training and validation loss, and validation AUC, per epoch.

    Args:
        history: per epoch metrics saved by train.py (history.csv).
        save_path: where to save the figure.
        death: plot death metrics (val_bce, val_auc).
        bad: plot bad track metrics (val_bad_bce, val_bad_auc).
    """
    epochs = history['epoch']
    best = history.loc[history['val_score'].idxmin(), 'epoch']

    plt.rcParams["font.family"] = 'serif'
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, history['train_loss'], color='black',
                 label='Train loss (total)')
    if death:
        ax_loss.plot(epochs, history['val_bce'], color='tab:blue',
                     label='Val death BCE')
        ax_auc.plot(epochs, history['val_auc'], color='tab:blue',
                    label='Val death AUC (per frame)')
    if bad:
        ax_loss.plot(epochs, history['val_bad_bce'], color='tab:red',
                     label='Val bad track BCE')
        ax_auc.plot(epochs, history['val_bad_auc'], color='tab:red',
                    label='Val bad track AUC (per cell)')
    for ax in (ax_loss, ax_auc):
        ax.axvline(best, color='grey', ls='--', lw=1, label=f'Best epoch ({best})')
        ax.set_xlabel('Epoch')
        ax.grid(True)
        ax.legend(fontsize='small')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Loss')
    ax_auc.set_ylabel('AUC')
    ax_auc.set_title('Validation AUC')
    fig.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_run(run_dir: Path, threshold: float = 0.5) -> None:
    """Make the plots relevant to a run's losses."""
    predictions = pd.read_csv(run_dir / 'best_val_predictions.csv')
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
    death = config.get('loss', 'frame') != 'none'
    bad = config.get('bad_weight', 1.0) > 0
    plot_history(pd.read_csv(run_dir / 'history.csv'), run_dir / 'loss.png',
                 death, bad)
    if bad:
        plot_bad_tracks(predictions, run_dir / 'bad_tracks.png', threshold)
    if death:
        plot_death_frames(predictions, run_dir / 'death_frames.png',
                          config.get('tolerance', 5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot validation predictions of a training run.')
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help='p_bad threshold for the confusion matrix.')
    args = parser.parse_args()
    plot_run(args.run_dir, args.threshold)
