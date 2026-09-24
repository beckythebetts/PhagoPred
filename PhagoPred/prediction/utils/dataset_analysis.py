"""
Dataset analysis utilities for experiment suites.
Analyses event/censoring counts and time distributions across all runs in an experiment.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from PhagoPred.survival_v2.configs.experiments import (MODELS, DATASETS,
                                                       FEATURE_COMBOS)
from PhagoPred.survival_v2.experiments.run_experiments import build_datasets


def analyse_suite_datasets(suite_dir: Path, save: bool = True) -> dict:
    """
    Iterate through all experiment runs in a suite directory, build the
    train/val datasets for each, and collect event/censoring statistics.

    Args:
        suite_dir: Path to the experiment suite results directory
                   (e.g. results/classical_comparison_20260217_105014/)
        save: If True, save the results JSON inside suite_dir.

    Returns:
        dict mapping experiment name -> {
            'dataset': str,
            'feature_combo': str,
            'model': str,
            'train': {'total': int, 'events': int, 'censored': int,
                      'event_times': list, 'censored_obs_times': list},
            'val':   { ... same ... }
        }
    """
    suite_dir = Path(suite_dir)
    experiment_dirs = sorted([d for d in suite_dir.iterdir() if d.is_dir()])

    # Cache datasets by (dataset_name, feature_combo, is_classical) to avoid
    # rebuilding the same dataset for different model/loss/attention combos.
    dataset_cache = {}
    results = {}

    for exp_dir in experiment_dirs:
        cfg_path = exp_dir / 'config.json'
        if not cfg_path.exists():
            continue

        with open(cfg_path, 'r') as f:
            config = json.load(f)

        dataset_name = config['dataset']
        feature_combo_name = config['feature_combo']
        model_name = config['model']
        # classical = is_classical_model(MODELS[model_name])
        classical = config.is_classical

        cache_key = (dataset_name, feature_combo_name, classical)

        if cache_key not in dataset_cache:
            dataset_config = DATASETS[dataset_name]
            feature_combo = FEATURE_COMBOS[feature_combo_name]
            train_ds, val_ds = build_datasets(dataset_config,
                                              feature_combo,
                                              is_classical=classical)
            dataset_cache[cache_key] = (train_ds, val_ds)

        train_ds, val_ds = dataset_cache[cache_key]

        stats = {}
        for split_name, ds in [('train', train_ds), ('val', val_ds)]:
            death_frames = ds.cell_metadata['Death Frames'].astype(float)
            start_frames = ds.cell_metadata['Start Frames'].astype(float)
            end_frames = ds.cell_metadata['End Frames'].astype(float)

            is_event = ~np.isnan(death_frames)
            is_censored = np.isnan(death_frames)

            obs_times = end_frames - start_frames + 1

            # Time-to-event for uncensored cells (death - start)
            event_times = (death_frames[is_event] -
                           start_frames[is_event]).tolist()
            # Observation duration for censored cells
            censored_obs_times = obs_times[is_censored].tolist()
            # Absolute event/censoring times (frame numbers, not relative)
            abs_event_times = death_frames[is_event].tolist()
            abs_censored_times = end_frames[is_censored].tolist()
            # Sequence lengths for all cells
            seq_lengths = obs_times.tolist()

            stats[split_name] = {
                'total': int(len(death_frames)),
                'events': int(is_event.sum()),
                'censored': int(is_censored.sum()),
                'event_times': event_times,
                'censored_obs_times': censored_obs_times,
                'abs_event_times': abs_event_times,
                'abs_censored_times': abs_censored_times,
                'seq_lengths': seq_lengths,
            }

        results[exp_dir.name] = {
            'dataset': dataset_name,
            'feature_combo': feature_combo_name,
            'model': model_name,
            **stats,
        }

    if save:
        out_path = suite_dir / 'dataset_analysis.json'
        # Save a compact version without the raw time lists for the summary
        summary = {}
        for name, info in results.items():
            summary[name] = {
                'dataset': info['dataset'],
                'feature_combo': info['feature_combo'],
                'model': info['model'],
                'train': {
                    k: v
                    for k, v in info['train'].items()
                    if k not in ('event_times', 'censored_obs_times')
                },
                'val': {
                    k: v
                    for k, v in info['val'].items()
                    if k not in ('event_times', 'censored_obs_times')
                },
            }
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved dataset analysis summary to {out_path}")

    return results


def plot_suite_event_distributions(
    suite_dir: Path,
    results: dict = None,
    bins: int = 30,
):
    """
    Plot event-time and censoring-time distributions for each unique
    dataset configuration in the suite.

    Produces one figure per unique (dataset, feature_combo) combination,
    with side-by-side train/val histograms.

    Args:
        suite_dir: Path to the experiment suite results directory.
        results: Pre-computed results dict from analyse_suite_datasets.
                 If None, will compute it.
        bins: Number of histogram bins.
    """
    suite_dir = Path(suite_dir)

    if results is None:
        results = analyse_suite_datasets(suite_dir, save=True)

    # Group experiments by (dataset, feature_combo) since those share data
    groups = {}
    for exp_name, info in results.items():
        key = (info['dataset'], info['feature_combo'])
        if key not in groups:
            groups[key] = info

    plots_dir = suite_dir / 'dataset_analysis_plots'
    plots_dir.mkdir(exist_ok=True)

    for (dataset_name, fc_name), info in groups.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, split in zip(axes, ['train', 'val']):
            split_info = info[split]
            event_times = np.array(split_info['event_times'])
            censored_times = np.array(split_info['censored_obs_times'])

            all_times = np.concatenate([event_times, censored_times]) \
                if len(censored_times) > 0 else event_times
            bin_edges = np.linspace(0, np.max(all_times), bins + 1) \
                if len(all_times) > 0 else np.linspace(0, 1, bins + 1)

            ax.hist(event_times,
                    bins=bin_edges,
                    alpha=0.7,
                    label=f'Events (n={len(event_times)})',
                    color='tab:red')
            if len(censored_times) > 0:
                ax.hist(censored_times,
                        bins=bin_edges,
                        alpha=0.7,
                        label=f'Censored (n={len(censored_times)})',
                        color='tab:blue')

            ax.set_xlabel('Time (frames)')
            ax.set_ylabel('Count')
            ax.set_title(
                f'{split.capitalize()} '
                f'({split_info["events"]}E / {split_info["censored"]}C)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Dataset: {dataset_name}  |  Features: {fc_name}',
                     fontsize=13,
                     fontweight='bold')
        fig.tight_layout()

        save_path = plots_dir / f'{dataset_name}_{fc_name}_event_dist.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

    print(f"\nAll distribution plots saved to {plots_dir}")


def plot_suite_absolute_distributions(
    suite_dir: Path,
    results: dict = None,
    bins: int = 30,
):
    """
    Plot absolute event/censoring frame distributions and sequence length
    distributions for each unique dataset configuration in the suite.

    Produces one figure per unique (dataset, feature_combo) with three rows:
      Row 1: Absolute event & censoring times (frame numbers)
      Row 2: Sequence length distributions
    """
    suite_dir = Path(suite_dir)

    if results is None:
        results = analyse_suite_datasets(suite_dir, save=True)

    groups = {}
    for exp_name, info in results.items():
        key = (info['dataset'], info['feature_combo'])
        if key not in groups:
            groups[key] = info

    plots_dir = suite_dir / 'dataset_analysis_plots'
    plots_dir.mkdir(exist_ok=True)

    for (dataset_name, fc_name), info in groups.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for col, split in enumerate(['train', 'val']):
            split_info = info[split]
            abs_event = np.array(split_info['abs_event_times'])
            abs_cens = np.array(split_info['abs_censored_times'])
            seq_lens = np.array(split_info['seq_lengths'])

            # --- Row 0: absolute event/censoring times ---
            ax = axes[0, col]
            all_abs = np.concatenate([abs_event, abs_cens]) \
                if len(abs_cens) > 0 else abs_event
            bin_edges = np.linspace(0, np.max(all_abs), bins + 1) \
                if len(all_abs) > 0 else np.linspace(0, 1, bins + 1)

            ax.hist(abs_event,
                    bins=bin_edges,
                    alpha=0.7,
                    label=f'Events (n={len(abs_event)})',
                    color='tab:red')
            if len(abs_cens) > 0:
                ax.hist(abs_cens,
                        bins=bin_edges,
                        alpha=0.7,
                        label=f'Censored (n={len(abs_cens)})',
                        color='tab:blue')
            ax.set_xlabel('Absolute frame')
            ax.set_ylabel('Count')
            ax.set_title(f'{split.capitalize()} — Absolute times')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # --- Row 1: sequence lengths ---
            ax = axes[1, col]
            ax.hist(seq_lens,
                    bins=bins,
                    alpha=0.7,
                    color='tab:green',
                    label=f'All cells (n={len(seq_lens)})')
            ax.set_xlabel('Sequence length (frames)')
            ax.set_ylabel('Count')
            ax.set_title(f'{split.capitalize()} — Sequence lengths')
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Dataset: {dataset_name}  |  Features: {fc_name}',
                     fontsize=13,
                     fontweight='bold')
        fig.tight_layout()

        save_path = plots_dir / f'{dataset_name}_{fc_name}_absolute_dist.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")

    print(f"\nAll absolute distribution plots saved to {plots_dir}")


def print_dataset_summary(suite_dir: Path):
    """Print a formatted table of event/censoring counts for a suite."""
    suite_dir = Path(suite_dir)
    json_path = suite_dir / 'dataset_analysis.json'

    if json_path.exists():
        with open(json_path, 'r') as f:
            summary = json.load(f)
    else:
        full_results = analyse_suite_datasets(suite_dir, save=True)
        summary = {}
        for name, info in full_results.items():
            summary[name] = {
                'dataset': info['dataset'],
                'model': info['model'],
                'train': {
                    k: v
                    for k, v in info['train'].items()
                    if k not in ('event_times', 'censored_obs_times')
                },
                'val': {
                    k: v
                    for k, v in info['val'].items()
                    if k not in ('event_times', 'censored_obs_times')
                },
            }

    header = (f"{'Experiment':<65} {'Train':>20}  {'Val':>20}")
    sub = (f"{'':65} {'Tot':>5} {'Ev':>5} {'Cen':>5}  "
           f"  {'Tot':>5} {'Ev':>5} {'Cen':>5}")
    print('=' * 95)
    print(header)
    print(sub)
    print('-' * 95)

    for exp_name, info in summary.items():
        tr = info['train']
        va = info['val']
        print(f"{exp_name:<65} "
              f"{tr['total']:>5} {tr['events']:>5} {tr['censored']:>5}  "
              f"  {va['total']:>5} {va['events']:>5} {va['censored']:>5}")

    print('=' * 95)


def _pmf_from_hazards(hazards: np.ndarray) -> np.ndarray:
    hazards = np.clip(hazards, 0.0, 1.0)
    sf = np.cumprod(1 - hazards)
    pmf = np.empty_like(hazards)
    pmf[0] = hazards[0]
    pmf[1:] = sf[:-1] * hazards[1:]
    return pmf


def _expected_time(pmf: np.ndarray) -> float:
    t = np.arange(len(pmf))
    total = pmf.sum()
    if total < 1e-12:
        return np.nan
    return np.sum(t * pmf) / total


def plot_rule_expected_time_shift(
    h5_path: Path,
    save: bool = True,
):
    """
    For each rule stored in the synthetic HDF5 file, compute how much it
    shifts the expected event time per cell, then plot the distributions.

    For every cell the function computes:
        E[T | all rules]  vs  E[T | all rules except rule_i]
        shift_i = E_without - E_with   (positive => rule brings death earlier)

    Args:
        h5_path: Path to the synthetic HDF5 dataset.
        save: If True, save the figure next to the HDF5 file.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, 'r') as f:
        grp = f['Cells/Phase']
        rule_names = [
            k.replace('HazardContribution_', '') for k in grp.keys()
            if k.startswith('HazardContribution_')
        ]
        contributions = {
            name: grp[f'HazardContribution_{name}'][:]
            for name in rule_names
        }

    # Reconstruct the raw (pre-smooth) total hazard from stored contributions
    sample_shape = next(iter(contributions.values())).shape
    T, N = sample_shape
    raw_total = np.zeros(sample_shape, dtype=np.float32)
    for contrib in contributions.values():
        raw_total += contrib

    # Expected time with all rules
    E_with = np.empty(N)
    for c in range(N):
        # h = gaussian_filter1d(raw_total[:, c], sigma=smooth_sigma)
        h = raw_total[:, c]
        h = np.clip(h, 0.0, 1.0)
        E_with[c] = _expected_time(_pmf_from_hazards(h))

    # Per-rule: expected time without rule_i
    shifts = {}
    for name, contrib in contributions.items():
        raw_without = raw_total - contrib
        E_without = np.empty(N)
        for c in range(N):
            # h = gaussian_filter1d(raw_without[:, c], sigma=smooth_sigma)
            h = raw_without[:, c]
            h = np.clip(h, 0.0, 1.0)
            E_without[c] = _expected_time(_pmf_from_hazards(h))
        shifts[
            name] = E_without - E_with  # positive => rule brings death earlier

    # --- Plotting ---
    n_rules = len(rule_names)
    fig, axes = plt.subplots(1,
                             n_rules,
                             figsize=(5 * n_rules, 5),
                             squeeze=False)
    axes = axes[0]

    for ax, name in zip(axes, rule_names):
        s = shifts[name]
        valid = s[~np.isnan(s)]
        ax.hist(valid, bins=40, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        median = np.median(valid) if len(valid) > 0 else 0
        ax.axvline(median,
                   color='tab:red',
                   linestyle='-',
                   linewidth=1.2,
                   label=f'median={median:.1f}')
        ax.set_xlabel('Shift in E[T] (frames)')
        ax.set_ylabel('Cell count')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Expected event-time shift per rule\n(positive = rule brings death earlier)',
        fontsize=13,
        fontweight='bold')
    fig.tight_layout()

    if save:
        save_path = h5_path.parent / f'{h5_path.stem}_rule_time_shifts.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {save_path}")
    else:
        plt.show()


if __name__ == '__main__':

    # For rule contribution analysis on synthetic data:
    h5_path_t = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/Datasets/synthetic_variants/baseline_train.h5'
    )
    h5_path_v = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/Datasets/synthetic_variants/baseline_val.h5'
    )

    plot_rule_expected_time_shift(h5_path_t)
    plot_rule_expected_time_shift(h5_path_v)
