from __future__ import annotations
from collections import defaultdict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from PhagoPred.utils.h5_file import H5File

CMAP = plt.get_cmap('Set1')


def group_colours(labels: list, colours: dict | None = None) -> list:
    """Colour per label, using CMAP as default and overriding with colours dict."""
    colours = {} if colours is None else {
        str(k): v
        for k, v in colours.items()
    }
    defaults = iter(CMAP(i) for i in range(CMAP.N))
    return [
        colours[str(label)] if str(label) in colours else next(defaults)
        for label in labels
    ]


def plot_mean_std(ax: plt.axes,
                  times: np.ndarray,
                  means: np.ndarray,
                  stds: np.ndarray,
                  label: str,
                  bar: bool,
                  idx: int = 0,
                  n_groups: int = 1,
                  color=None) -> None:
    """One group's mean +- std over wells, as a line and band, or as bars
    side by side with the other groups.
    """
    color = CMAP(idx) if color is None else color
    if bar:
        spacing = np.min(np.diff(times)) if len(times) > 1 else 1.0
        width = 0.8 * spacing / n_groups
        offset = (idx - (n_groups - 1) / 2) * width
        ax.bar(times + offset,
               means,
               width=width,
               yerr=stds,
               capsize=3,
               label=label,
               color=color,
               edgecolor='none')

    else:
        ax.plot(times, means, label=label, color=color)
        ax.fill_between(times,
                        means - stds,
                        means + stds,
                        alpha=0.5,
                        color=color,
                        edgecolor='none')


def plot_by_label(
    label_to_files: dict[str | int, list[Path]],
    well_values: Callable[[Path], tuple[np.ndarray, np.ndarray]],
    ylabel: str,
    save_as: Path,
    bar: bool = False,
    label_prefix: str = 'MOI ',
    label_suffix: str = "",
    colours: dict | None = None,
    start_at: int = 0,
) -> None:
    """Mean +- standard error over each group's wells, one line (or set of
    bars) per group.

    Args:
        label_to_files: group label -> its wells' h5 files.
        well_values: h5 file -> (times / hours, values) for one well; every
            well in a group must give the same times.
        ylabel: y axis label.
        save_as: where to save the figure.
        bar: bars side by side per group, instead of lines and bands.
        label_prefix, label_suffix: put around each group label in the
            legend.
        colours: group label -> colour for chosen groups, e.g. {0: 'black'}
            for a control; others take CMAP colours in order.
        start_at: leave out this many points at the start.
    """
    plt.rcParams["font.family"] = 'serif'
    fig, ax = plt.subplots()
    labels = list(label_to_files)
    try:
        # Numeric groups (MOI, infection hours, ...) plotted in order
        labels = sorted(labels, key=float)
    except ValueError:
        pass
    colours = group_colours(labels, colours)
    for idx, label in enumerate(labels):
        wells = [
            well_values(file)
            for file in tqdm(label_to_files[label], desc=f'{label}')
        ]
        times = wells[0][0]
        if not all(np.array_equal(t, times) for t, _ in wells):
            raise ValueError(f'{label}: wells have different time points')
        values = np.array([v for _, v in wells], dtype=float)
        means = np.nanmean(values, axis=0)
        sems = np.nanstd(values, axis=0) / np.sqrt(len(values))
        plot_mean_std(
            ax,
            times[start_at:],
            means[start_at:],
            sems[start_at:],
            f'{label_prefix}{label}{label_suffix}',
            bar,
            idx,
            len(labels),
            color=colours[idx],
        )

    ax.set_xlabel('Time / hours')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    plt.savefig(save_as)


def frame_time(h5_file: Path) -> float:
    """Hours per frame, from the file's imaging metadata."""
    # return H5File(h5_file).frame_time
    return 0.25


def label_to_files_mapping(data_path: Path) -> dict:
    with open(data_path, 'r') as f:
        label_to_files = json.load(f)
        label_to_files = {
            str(k): [data_path.parent / f'{str(file)}.h5' for file in v]
            for k, v in label_to_files.items()
        }
    return label_to_files


def generate_files_mapping(mapping: dict[str | int, str],
                           h5_dir: Path) -> dict[str, Path]:
    """
    generate a file mapping from a dict of {label: str expression in fielname}
    """
    file_mapping = defaultdict(list)
    for file in h5_dir.glob('*.h5'):
        for k, v in mapping.items():
            if v in file.stem:
                file_mapping[k].append(file)
    return file_mapping
