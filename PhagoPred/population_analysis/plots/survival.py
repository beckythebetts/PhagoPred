"""Alive cells, death rates, survival curves and event times per group."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PhagoPred.utils.h5_file import H5File
from ._common import (group_colours, plot_val_range, plot_by_label, frame_time,
                      alive_and_deaths)


def _plot_event_time_distribution(ax: plt.axes,
                                  label: str,
                                  h5_files: list[Path],
                                  per_alive_cell: bool = False,
                                  num_bins: int = 20,
                                  color=None) -> None:
    """Histogram of event times over a group's wells.

    per_alive_cell divides each bin's events by the cell-hours alive in it
    (summed over wells), giving events per alive cell per hour, so fewer
    events because fewer cells are alive isn't read as a lower rate. The
    title reports events per well and the overall rate per cell-hour.
    """
    hours_per_frame = frame_time(h5_files[0])
    num_frames = H5File(h5_files[0]).num_frames
    frames = np.arange(num_frames)
    edges = np.linspace(0, num_frames, num_bins + 1)

    events = np.zeros(num_bins)
    cell_hours = np.zeros(num_bins)
    for h5_file in tqdm(h5_files, desc=f'Getting event times for {label}'):
        event_frames = H5File(h5_file).event_frames
        events += np.histogram(event_frames[~np.isnan(event_frames)], edges)[0]
        alive, _ = alive_and_deaths(h5_file, frames)
        cell_hours += np.histogram(frames, edges,
                                   weights=alive)[0] * hours_per_frame

    with np.errstate(invalid='ignore', divide='ignore'):
        heights = events / cell_hours if per_alive_cell else events
    ax.stairs(heights, edges * hours_per_frame, fill=True, color=color)
    ax.set_title(
        f'MOI {label}: {events.sum() / len(h5_files):.1f} events '
        f'per well, {events.sum() / cell_hours.sum():.2e} per cell-hour',
        fontsize=9)


def plot_event_time_distributions(
    label_to_files: dict[str, list[Path]],
    save_as: Path = Path('temp') / 'event_time_distribution.png',
    per_alive_cell: bool = False,
    num_bins: int = 20,
    colours: dict | None = None,
) -> None:
    """Event time histograms, one panel per group.

    Args:
        per_alive_cell: show events per alive cell per hour in each bin,
            instead of event counts summed over wells.
        num_bins: histogram bins over the whole time course.
        colours: group label -> colour for chosen groups; others take CMAP
            colours in order.
    """
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(len(label_to_files),
                            1,
                            figsize=(4, 2 * len(label_to_files)),
                            sharex=True,
                            sharey=True,
                            squeeze=False)
    colours = group_colours(list(label_to_files), colours)
    for ax, (label, files), colour in zip(axs[:, 0], label_to_files.items(),
                                          colours):
        _plot_event_time_distribution(ax, label, files, per_alive_cell,
                                      num_bins, colour)
        ax.set_ylabel('Events / cell-hour' if per_alive_cell else 'Events',
                      fontsize=8)
    axs[-1, 0].set_xlabel('Time / hours')
    fig.tight_layout()
    plt.savefig(save_as)


def plot_manual_alive_counts(label_to_counts: dict[str, Path],
                             save_as: Path = Path('temp') / 'counts.png',
                             frame_time: float = 0.25,
                             bar: bool = False,
                             death_rate: bool = False) -> None:
    fig, ax = plt.subplots()
    for i, (label, counts_txt) in enumerate(label_to_counts.items()):
        _plot_manual_alive_count(
            ax,
            label,
            counts_txt,
            frame_time,
            death_rate,
            bar,
            i,
            len(label_to_counts),
        )
    ax.set_xlabel('Time / hours')
    ax.set_ylabel('Death rate / hour$^{-1}$'
                  if death_rate else 'Alive cells count per well')
    ax.legend()
    ax.grid()
    plt.savefig(save_as)


def _plot_manual_alive_count(ax: plt.axes,
                             label: str,
                             counts_txt: Path,
                             frame_time: float,
                             death_rate: bool,
                             bar: bool,
                             idx: int = 0,
                             n_groups: int = 1) -> None:
    samples = []
    with open(counts_txt, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            if line[0] == '#':
                continue
            _, frame, count, _ = line.strip().split('|')
            frame, count = int(frame), int(count)

            if not samples or frame <= samples[-1][-1][0]:
                samples.append([])
            samples[-1].append((frame, count))

    values = defaultdict(list)
    for sample in samples:
        sample_frames, sample_counts = np.array(sample, dtype=float).T
        if death_rate:
            dt = np.diff(sample_frames) * frame_time
            rates = -np.diff(sample_counts) / dt / (
                (sample_counts[:-1] + sample_counts[1:]) / 2)
            mid_frames = (sample_frames[:-1] + sample_frames[1:]) / 2
            for frame, rate in zip(mid_frames, rates):
                values[frame].append(rate)
        else:
            for frame, count in zip(sample_frames, sample_counts):
                values[frame].append(count)

    frames = np.array(sorted(values))
    means = np.array([np.mean(values[f]) for f in frames])
    stds = np.array([np.std(values[f]) for f in frames])
    plot_val_range(ax, frames * frame_time, means, stds, f'MOI {label}', bar,
                   idx, n_groups)


def _well_alive(h5_file: Path,
                every: int,
                death_rate: bool,
                survival_curve: bool,
                start: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """One well's alive cells at every every'th frame, or its deaths per
    cell-hour alive between them (timed at the middle of each interval)."""
    hours_per_frame = frame_time(h5_file)
    num_frames = H5File(h5_file).num_frames
    frames = np.arange(start, num_frames, every)
    alive, deaths = alive_and_deaths(h5_file, frames)
    if not death_rate:
        if survival_curve:
            return frames * hours_per_frame, np.log(alive)
        return frames * hours_per_frame, alive

    alive_each_frame, _ = alive_and_deaths(h5_file, np.arange(num_frames))
    cell_hours = np.add.reduceat(alive_each_frame,
                                 frames)[:-1] * hours_per_frame
    with np.errstate(invalid='ignore', divide='ignore'):
        rates = deaths / cell_hours
    return (frames[:-1] + frames[1:]) / 2 * hours_per_frame, rates


def _reacquired(h5: H5File,
                lost: np.ndarray,
                max_gap: int = 20,
                max_dist: float = 100) -> np.ndarray:
    """Which lost tracks are followed by a new track on the same cell: one
    alive at its start, starting from 5 frames before to max_gap frames
    after the loss, within max_dist pixels of where the cell was lost."""
    first, last, death = h5.first_frames, h5.last_frames, h5.death_frames
    coords = h5.get_features(['X', 'Y'])
    X, Y = coords['X'], coords['Y']
    cells = np.arange(len(first))
    starts = np.nan_to_num(first).astype(int)
    start_xy = np.stack([X[starts, cells], Y[starts, cells]], axis=1)
    starts_alive = ~np.isnan(first) & ~(death <= first + 2)

    reacquired = np.zeros_like(lost)
    for c in np.flatnonzero(lost):
        lost_at = int(last[c])
        new = starts_alive & (first > lost_at - 5) & (first
                                                      <= lost_at + max_gap)
        new[c] = False
        dists = np.hypot(*(start_xy[new] - [X[lost_at, c], Y[lost_at, c]]).T)
        reacquired[c] = (dists < max_dist).any()
    return reacquired


def _censoring_vs_death(
        h5_file: Path,
        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One well's deaths, mid-experiment censorings (tracks lost before a
    death is seen) and those of them never picked up again by a new track,
    binned by frame, and the number censored by the experiment ending. Bad
    tracks and cells dead from the start are left out."""
    h5 = H5File(h5_file)
    first, last, death = h5.first_frames, h5.last_frames, h5.death_frames
    good = ~np.isnan(last)
    dies = good & (death > first)
    censored = good & np.isnan(death)
    at_end = censored & (last == h5.num_frames - 1)
    lost = censored & ~at_end
    never_found = lost & ~_reacquired(h5, lost)
    return (np.histogram(death[dies],
                         edges)[0], np.histogram(last[lost], edges)[0],
            np.histogram(last[never_found], edges)[0], at_end.sum())


# def _kaplan_meier(h5_file: Path)


def plot_censoring_vs_death(
    label_to_files: dict[str, list[Path]],
    save_as: Path = Path('temp') / 'censoring_vs_death.png',
    num_bins: int = 20,
    label_prefix: str = 'MOI ',
) -> None:
    """Death and censoring time histograms (summed over wells), one panel
    per group. Censored tracks never picked up again by a new track are
    shown separately, as those are the ones that could be missed deaths."""
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(len(label_to_files),
                            1,
                            figsize=(6.5, 2 * len(label_to_files)),
                            sharex=True,
                            squeeze=False)
    for ax, (label, files) in zip(axs[:, 0], label_to_files.items()):
        hours_per_frame = frame_time(files[0])
        edges = np.linspace(0, H5File(files[0]).num_frames, num_bins + 1)
        deaths, censored, never_found = (np.zeros(num_bins) for _ in range(3))
        at_end = 0
        for h5_file in tqdm(files, desc=f'Censoring vs death for {label}'):
            d, c, n, e = _censoring_vs_death(h5_file, edges)
            deaths, censored, never_found, at_end = (deaths + d, censored + c,
                                                     never_found + n,
                                                     at_end + e)
        ax.stairs(deaths, edges * hours_per_frame, label='Deaths')
        ax.stairs(censored, edges * hours_per_frame, label='Censored')
        ax.stairs(never_found,
                  edges * hours_per_frame,
                  label='Censored, not picked up again',
                  linestyle='--')
        ax.set_title(
            f'{label_prefix}{label}: {at_end} censored at end of experiment',
            fontsize=9)
        ax.set_ylabel('Cells', fontsize=8)
    axs[0, 0].legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1))
    axs[-1, 0].set_xlabel('Time / hours')
    fig.tight_layout()
    plt.savefig(save_as)


def plot_num_alive_cells(label_to_files: dict[str, list[Path]],
                         save_as: Path = Path('temp') / 'alive_cells.png',
                         every: int = 1,
                         death_rate: bool = False,
                         survival_curve: bool = False,
                         **kwargs) -> None:
    plot_by_label(label_to_files,
                  partial(_well_alive,
                          every=every,
                          death_rate=death_rate,
                          survival_curve=survival_curve),
                  ylabel=('Death rate / hour$^{-1}$'
                          if death_rate else 'Alive cells per well'),
                  save_as=save_as,
                  **kwargs)
