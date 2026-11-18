from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.h5_file import H5File
from ._common import CMAP, generate_files_mapping


def manual_count(h5_file: Path, frame: int = 0) -> float:
    """The manual cell count saved by the GUI as num_cells_<frame>, or NaN if
    this file doesn't have one."""
    count = H5File(h5_file).get_attr(f'num_cells_{frame}')
    return np.nan if count is None else float(count)


def _sem(values: np.ndarray) -> float:
    return np.nanstd(values, ddof=1) / np.sqrt(np.sum(~np.isnan(values)))


def plot_manual_counts_by_day(
    day_dirs: dict[str, Path],
    conditions: dict[str, str | list[str]],
    save_as: Path = Path('temp') / 'manual_counts_by_day.png',
    frame: int = 0,
) -> dict:
    """Manual cell counts per condition on each day, and how they change.

    Left: counts per well (dots) with mean +- standard error (bars), one
    group of bars per condition, one bar per day. Right: the last day's mean
    count as a fraction of the first day's, per condition, with the standard
    error propagated from both days. Wells aren't paired between days, so
    this compares means only.

    Args:
        day_dirs: day label -> folder of that day's .h5 files, in time order,
            e.g. {'Day 0': Path(...), 'Day 3': Path(...)}.
        conditions: condition label -> text (or list of texts) that appears
            in the file names of that condition's wells, as for
            generate_files_mapping.
        frame: which frame's saved count (num_cells_<frame>) to use.

    Returns:
        {condition: {day: counts of its wells}}, NaN for missing counts.
    """
    plt.rcParams["font.family"] = 'serif'
    days = list(day_dirs)
    counts = {c: {} for c in conditions}
    for day, folder in day_dirs.items():
        # generate_files_mapping splits a bare string into characters
        files = generate_files_mapping(
            {
                c: [m] if isinstance(m, str) else m
                for c, m in conditions.items()
            }, Path(folder))
        for condition in conditions:
            wells = sorted(set(files.get(condition, [])))
            values = np.array([manual_count(f, frame) for f in wells])
            for f, v in zip(wells, values):
                if np.isnan(v):
                    print(f'{f.name}: no num_cells_{frame}, skipped')
            counts[condition][day] = values
            if len(wells) == 0:
                print(f'{day}/{condition}: no files found')

    fig, (ax, ax_ratio) = plt.subplots(1,
                                       2,
                                       figsize=(11, 4.5),
                                       gridspec_kw={'width_ratios': [2, 2]})
    x = np.arange(len(conditions))
    width = 0.8 / len(days)
    rng = np.random.default_rng(0)
    for d, day in enumerate(days):
        offset = (d - (len(days) - 1) / 2) * width
        means = [np.nanmean(counts[c][day]) for c in conditions]
        sems = [_sem(counts[c][day]) for c in conditions]
        ax.bar(x + offset,
               means,
               width=width,
               yerr=sems,
               capsize=3,
               color=CMAP(d),
               alpha=0.6,
               label=day)
        for i, c in enumerate(conditions):
            v = counts[c][day]
            ax.scatter(x[i] + offset + rng.uniform(-0.3, 0.3, len(v)) * width,
                       v,
                       color=CMAP(d),
                       edgecolor='k',
                       linewidth=0.5,
                       zorder=3,
                       s=18)
    ax.set_xticks(x, list(conditions))
    ax.set_ylabel(f'Manual cell count per well (frame {frame})')
    ax.legend()
    ax.grid(axis='y')

    first, last = days[0], days[-1]
    ratios, errs = [], []
    for c in conditions:
        m0, m1 = np.nanmean(counts[c][first]), np.nanmean(counts[c][last])
        s0, s1 = _sem(counts[c][first]), _sem(counts[c][last])
        ratio = m1 / m0
        ratios.append(ratio)
        # Relative errors of independent means add in quadrature
        errs.append(ratio * np.hypot(s0 / m0, s1 / m1))
    ax_ratio.bar(x, ratios, yerr=errs, capsize=3, color='grey', alpha=0.6)
    ax_ratio.axhline(1, color='k', linestyle='--', linewidth=1)
    ax_ratio.set_xticks(x, list(conditions))
    ax_ratio.set_ylabel(f'{last} count / {first} count')
    ax_ratio.grid(axis='y')

    fig.tight_layout()
    Path(save_as).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_as)

    print(f'{"condition":<20}' + ''.join(f'{d:>22}' for d in days))
    for c in conditions:
        print(f'{c:<20}' + ''.join(
            f'{np.nanmean(counts[c][d]):>11.1f} ± {_sem(counts[c][d]):<5.1f}'
            f'(n={np.sum(~np.isnan(counts[c][d]))})' for d in days))
    return counts
