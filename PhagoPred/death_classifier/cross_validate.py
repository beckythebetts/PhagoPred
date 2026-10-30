"""K-fold cross validation of DeathNet.

Trains one model per fold (each cell is a val cell in exactly one fold),
evaluates each fold's last checkpoint on its val cells, then combines the
out of fold predictions so every labelled cell has one prediction from a
model that never trained on it.

The main result uses last.pt (a fixed number of epochs chosen in advance).
best.pt is chosen on the fold's own val cells, so its combined results are
optimistic and reported for comparison only.

Folds that already finished are skipped, so an interrupted run can be
restarted with the same command.

Usage (any train.py flags can be added):
    python -m PhagoPred.death_classifier.cross_validate --n-folds 5 \
        --labels temp/death_labelling_02_10/deaths.json \
        --out-dir temp/death_classifier/cv_window64_ds2 \
        --loss frame --bad-weight 0 --downsample 2 --epochs 40
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from PhagoPred.death_classifier.evaluate import evaluate_checkpoint
from PhagoPred.death_classifier.plots import plot_bad_tracks, plot_death_frames
from PhagoPred.death_classifier.config import TrainConfig
from PhagoPred.death_classifier.train import _parse_args, train


def death_metrics(predictions: pd.DataFrame, tolerance: int = 5) -> dict:
    """Death frame accuracy over good tracks."""
    good = predictions[~predictions['bad'].astype(bool)]
    true, predicted = good['true'], good['predicted']
    both_die = true.notna() & predicted.notna()
    errors = (predicted - true)[both_die].abs()
    return {
        'cells': int(len(good)),
        'within_5': int(good['correct'].astype(bool).sum()),
        'within_10_both_die': int((errors <= 10).sum()),
        'within_20_both_die': int((errors <= 20).sum()),
        'both_die': int(both_die.sum()),
        'median_error_both_die':
        float(errors.median()) if len(errors) else float('nan'),
        'state_correct': float((true.isna() == predicted.isna()).mean()),
        'alive_correct': int((true.isna() & predicted.isna()).sum()),
        'alive': int(true.isna().sum()),
    }


def bad_metrics(predictions: pd.DataFrame,
                thresholds: tuple[float, ...] = (0.3, 0.5)) -> dict:
    """Bad track detection over all cells."""
    is_bad = predictions['bad'].astype(bool)
    p_bad = predictions['p_bad']
    metrics = {
        'cells': int(len(predictions)),
        'bad': int(is_bad.sum()),
        'auc':
        float(roc_auc_score(is_bad, p_bad))
        if 0 < is_bad.sum() < len(is_bad) else float('nan'),
    }
    for threshold in thresholds:
        predicted = p_bad > threshold
        metrics[f'at_{threshold}'] = {
            'correct': int((predicted == is_bad).sum()),
            'bad_found': int((predicted & is_bad).sum()),
            'good_flagged_bad': int((predicted & ~is_bad).sum()),
        }
    return metrics


def _fold_done(fold_dir: Path, epochs: int) -> bool:
    history = fold_dir / 'history.csv'
    return (fold_dir / 'last.pt').exists() and history.exists() and len(
        pd.read_csv(history)) >= epochs


def cross_validate(base: TrainConfig, n_folds: int = 5) -> None:
    out_dir = base.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(n_folds):
        fold_dir = out_dir / f'fold_{fold}'
        cfg = replace(base, n_folds=n_folds, fold=fold, out_dir=fold_dir)
        if _fold_done(fold_dir, cfg.epochs):
            print(f'Fold {fold} already trained, skipping')
        else:
            print(f'\n=== Fold {fold + 1}/{n_folds} ===')
            train(cfg)
            torch.cuda.empty_cache()
        if not (fold_dir / 'last_val_predictions.csv').exists():
            evaluate_checkpoint(fold_dir, 'last.pt')

    summary = {}
    for checkpoint in ('last', 'best'):
        folds = []
        for fold in range(n_folds):
            predictions = pd.read_csv(out_dir / f'fold_{fold}' /
                                      f'{checkpoint}_val_predictions.csv')
            predictions['fold'] = fold
            folds.append(predictions)
        combined = pd.concat(folds, ignore_index=True)
        combined.to_csv(out_dir / f'cv_{checkpoint}_predictions.csv',
                        index=False)
        summary[checkpoint] = {}
        if base.loss != 'none':
            plot_death_frames(combined,
                              out_dir / f'cv_{checkpoint}_death_frames.png',
                              base.tolerance)
            summary[checkpoint] = {
                'all_folds': death_metrics(combined, base.tolerance),
                'per_fold':
                [death_metrics(f, base.tolerance) for f in folds],
            }
        if base.bad_weight > 0:
            plot_bad_tracks(combined,
                            out_dir / f'cv_{checkpoint}_bad_tracks.png', 0.3)
            summary[checkpoint]['bad_tracks'] = {
                'all_folds': bad_metrics(combined),
                'per_fold': [bad_metrics(f) for f in folds],
            }

    with open(out_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    for checkpoint, note in (('last', 'main result'),
                             ('best', 'optimistic, epoch chosen on val')):
        if 'bad_tracks' in summary[checkpoint]:
            m = summary[checkpoint]['bad_tracks']['all_folds']
            per_fold = [
                p['auc'] for p in summary[checkpoint]['bad_tracks']['per_fold']
            ]
            print(f"\n{checkpoint}.pt ({note}), bad tracks, all {m['cells']} "
                  f"cells ({m['bad']} bad): AUC {m['auc']:.3f} (per fold "
                  f"{', '.join(f'{p:.3f}' for p in per_fold)})")
            for threshold in (0.3, 0.5):
                t = m[f'at_{threshold}']
                print(f"  p_bad > {threshold}: {t['correct']}/{m['cells']} "
                      f"correct, {t['bad_found']}/{m['bad']} bad found, "
                      f"{t['good_flagged_bad']} good flagged bad")
        if 'all_folds' not in summary[checkpoint]:
            continue
        m = summary[checkpoint]['all_folds']
        per_fold = [p['within_5'] / p['cells']
                    for p in summary[checkpoint]['per_fold']]
        print(f"\n{checkpoint}.pt ({note}), all {m['cells']} cells:")
        print(f"  within ±{base.tolerance} frames: {m['within_5']}/{m['cells']} "
              f"({m['within_5'] / m['cells']:.0%}; per fold "
              f"{', '.join(f'{p:.0%}' for p in per_fold)})")
        print(f"  where both die: within ±10 {m['within_10_both_die']}/"
              f"{m['both_die']}, within ±20 {m['within_20_both_die']}/"
              f"{m['both_die']}, median error "
              f"{m['median_error_both_die']:.1f} frames")
        print(f"  alive/dead state correct {m['state_correct']:.1%}, alive "
              f"cells correct {m['alive_correct']}/{m['alive']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--n-folds', type=int, default=5)
    args, train_argv = parser.parse_known_args()
    cross_validate(_parse_args(train_argv), args.n_folds)
