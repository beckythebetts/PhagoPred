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
LINESTYLES = ['-', '--', ':', '-.']
HATCHES = ['', '//', '..', 'xx']


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


def running_mean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mean over all frames up to and including each one, ignoring NaN.
    NaN where nothing has been seen yet, and where the cell isn't tracked."""
    seen = ~np.isnan(values)
    total = np.cumsum(np.where(seen, values, 0), axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        means = total / np.cumsum(seen, axis=axis)
    return np.where(seen, means, np.nan)


def alive_mask(h5_file: Path, frames: np.ndarray) -> np.ndarray:
    """(len(frames), cells): tracked (First Frame <= t <= Last Frame) and
    before Death Frame. Likely bad tracks are False."""
    h5 = H5File(h5_file)
    first, last, death = h5.first_frames, h5.last_frames, h5.death_frames
    # NaN (bad tracks, alive cells' Death Frame) compares False throughout
    t = frames[:, None]
    return (first <= t) & (t <= last) & ~(death <= t)


def alive_and_deaths(h5_file: Path, frames: np.ndarray) -> tuple:
    """Alive cells at each of frames, and deaths between consecutive frames.

    A cell is alive while tracked (First Frame <= t <= Last Frame) and before
    its Death Frame. Deaths are only those during a track: cells dead from
    the start of their track (Death Frame == First Frame) were never seen
    alive, and a track ending isn't a death. Likely bad tracks are left out.

    Returns:
        (len(frames),) alive counts, (len(frames) - 1,) death counts.
    """
    h5 = H5File(h5_file)

    alive = alive_mask(h5_file, frames).sum(axis=1)
    first, last, death = h5.first_frames, h5.last_frames, h5.death_frames
    # # NaN (bad tracks, alive cells' Death Frame) compares False throughout
    # t = frames[:, None]
    # tracked = (first <= t) & (t <= last)
    # alive = (tracked & ~(death <= t)).sum(axis=1)

    dies_in_track = death > first
    deaths = (dies_in_track & (death >= frames[:-1, None]) &
              (death < frames[1:, None])).sum(axis=1)
    # deaths = -np.diff(alive)
    return alive, deaths


def plot_val_range(ax: plt.axes,
                   times: np.ndarray,
                   vals: np.ndarray,
                   mins: np.ndarray,
                   maxs: np.ndarray,
                   label: str,
                   bar: bool,
                   idx: int = 0,
                   n_groups: int = 1,
                   color=None,
                   linestyle: str = '-',
                   hatch: str = '') -> None:
    """One group's mean +- std over wells, as a line and band, or as bars
    side by side with the other groups.
    """
    color = CMAP(idx) if color is None else color
    if bar:
        spacing = np.min(np.diff(times)) if len(times) > 1 else 1.0
        width = 0.8 * spacing / n_groups
        offset = (idx - (n_groups - 1) / 2) * width
        ax.bar(times + offset,
               vals,
               width=width,
               yerr=(vals - min, maxs - vals),
               capsize=3,
               label=label,
               color=color,
               edgecolor='white',
               linewidth=0,
               hatch=hatch)

    else:
        ax.plot(times, vals, label=label, color=color)
        ax.fill_between(
            times,
            mins,
            maxs,
            alpha=0.5,
            color=color,
            edgecolor='none',
            linestyle=linestyle,
        )


def plot_by_label(
    label_to_files: dict[str | int, list[Path]],
    well_values: Callable[[Path], tuple[np.ndarray, np.ndarray | dict]],
    ylabel: str,
    save_as: Path,
    bar: bool = False,
    label_prefix: str = 'MOI ',
    label_suffix: str = "",
    colours: dict | None = None,
    start_at: int = 0,
    log_y: bool = False,
    percentile: float | None = None,
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
        wells = [v if isinstance(v, dict) else {"": v} for _, v in wells]
        keys = list(wells[0])
        for k, key in enumerate(keys):
            values = np.array([w[key] for w in wells], dtype=float)
            if percentile is None:
                vals = np.nanmean(values, axis=0)
                sems = np.nanstd(values, axis=0) / np.sqrt(len(values))
                mins, maxs = vals - sems, vals + sems
            else:
                vals = np.nanmedian(values, axis=0)
                mins = np.nanpercentile(values, percentile, axis=0)
                maxs = np.nanpercentile(values, 1.0 - percentile, axis=0)
            _label = '(mean ± 1 standard error)' if percentile is None else f'(median ({percentile}th-{percentile}th percentile))'
            plot_val_range(
                ax,
                times[start_at:],
                vals[start_at:],
                mins[start_at:],
                maxs[start_at:],
                f'{label_prefix}{label}{label_suffix} {key} {_label}'.rstrip(),
                bar,
                idx,
                len(labels),
                color=colours[idx],
                linestyle=LINESTYLES[k],
                hatch=HATCHES[k],
            )

    if log_y:
        ax.set_yscale('log')
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
            v = list(v)
            for v_i in v:
                if v_i in file.stem:
                    file_mapping[k].append(file)
    return file_mapping
