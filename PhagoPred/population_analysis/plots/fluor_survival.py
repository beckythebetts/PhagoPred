"""Hazard of death as a function of a cell's current bacterial burden."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from PhagoPred.utils.h5_file import H5File
from ._common import group_colours, frame_time, running_mean

LINESTYLES = ['-', '--', ':', '-.']


def _at_risk(
    h5_file: Path,
    lag: int = 0,
    start: int = 5,
    infected_thresh: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Burden lag frames earlier, whether the cell dies by the next frame,
    and the frame, for each cell-frame at risk in one well.

    A cell is at risk at frame t if it's alive and its track continues to
    t + 1 (so its fate at t + 1 is seen). Cells dead from the start of their
    track are never at risk; bad tracks are left out.
    """
    h5 = H5File(h5_file)
    first, last, death = h5.first_frames, h5.last_frames, h5.death_frames
    death = np.fmin(death, h5.event_frames)
    burden = h5.get_features('Fluor Total')
    if lag > 0 and infected_thresh is None:
        burden = np.concatenate(
            [np.full((lag, burden.shape[1]), np.nan), burden[:-lag]])
    if infected_thresh is not None:
        tracked = ~np.isnan(burden)
        infected = np.logical_or.accumulate(burden > infected_thresh, axis=0)
        burden = np.where(tracked, infected, np.nan)
    t = np.arange(h5.num_frames)[:, None]
    at_risk = ((first <= t) & (t < last) & ~(death <= t) & (t >= start)
               & ~np.isnan(burden))
    dies = death == t + 1
    frames = np.broadcast_to(t, burden.shape)
    return burden[at_risk], dies[at_risk], frames[at_risk]


def _burden_edges(burdens: np.ndarray, n_bins: int) -> np.ndarray:
    """A bin for zero burden, then quantile bins of the positive burdens."""
    positive = burdens[burdens > 0]
    return np.unique(
        np.concatenate([[0, np.nextafter(0, 1)],
                        np.quantile(positive, [0, 0.8, 0.99])]))


def plot_hazard_by_infection(
    label_to_files: dict[str | int, list[Path]],
    infected_thresh: float,
    save_as: Path = Path('temp') / 'hazard_by_infection.png',
    lag: int = 2,
    start: int = 96,
    events_as_deaths: bool = True,
    label_prefix: str = 'MOI ',
) -> None:
    """Deaths per cell-hour at risk for cells not yet infected vs infected
    (above infected_thresh at any frame up to lag frames earlier), one pair
    of bars per group. Poisson error bars."""
    hours_per_frame = frame_time(next(iter(label_to_files.values()))[0])
    edges = np.array([-0.5, 0.5, 1.5])
    labels = list(label_to_files)
    hazards, errs = np.zeros((len(labels), 2)), np.zeros((len(labels), 2))
    for i, label in enumerate(labels):
        wells = [
            _at_risk(f, lag, start, infected_thresh)
            for f in tqdm(label_to_files[label], desc=f'{label}')
        ]
        status, dies, _ = (np.concatenate(x) for x in zip(*wells))
        exposure = np.histogram(status, edges)[0] * hours_per_frame
        deaths = np.histogram(status[dies], edges)[0]
        print(f'{label_prefix}{label}: cell-hours {exposure}, deaths {deaths}')
        with np.errstate(invalid='ignore', divide='ignore'):
            hazards[i] = deaths / exposure
            errs[i] = np.sqrt(deaths) / exposure

    plt.rcParams["font.family"] = 'serif'
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    for k, (name, hatch) in enumerate([('Not yet infected', ''),
                                       ('Infected', '//')]):
        ax.bar(x + (k - 0.5) * 0.4,
               hazards[:, k],
               width=0.4,
               yerr=errs[:, k],
               capsize=3,
               hatch=hatch,
               label=name,
               color=group_colours(labels),
               edgecolor='white',
               linewidth=0)
    ax.set_xticks(x, [f'{label_prefix}{label}' for label in labels])
    ax.set_ylabel('Deaths / cell-hour at risk')
    ax.legend()
    fig.tight_layout()
    plt.savefig(save_as)


def plot_hazard_vs_burden(
    label_to_files: dict[str | int, list[Path]],
    save_as: Path = Path('temp') / 'hazard_vs_burden.png',
    n_burden_bins: int = 1,
    n_time_windows: int = 1,
    lag: int = 24,
    start: int = 96,
    label_prefix: str = 'MOI ',
    label_suffix: str = "",
    colours: dict | None = None,
) -> None:
    """Deaths per cell-hour at risk against burden (Fluor Total lag frames
    earlier), one line per group, split by time window with linestyles.

    Error bars are Poisson (sqrt(deaths) / exposure), pooling cell-frames
    over a group's wells, so they ignore variation between wells.
    """
    groups = {
        label: [_at_risk(f, lag, start) for f in tqdm(files, desc=f'{label}')]
        for label, files in label_to_files.items()
    }
    groups = {
        label: tuple(np.concatenate(x) for x in zip(*wells))
        for label, wells in groups.items()
    }
    edges = _burden_edges(np.concatenate([b for b, _, _ in groups.values()]),
                          n_burden_bins)
    first_files = [files[0] for files in label_to_files.values()]
    hours_per_frame = frame_time(first_files[0])
    time_edges = np.linspace(start,
                             H5File(first_files[0]).num_frames,
                             n_time_windows + 1)

    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(1,
                            n_time_windows,
                            figsize=(4 * n_time_windows, 3.5),
                            sharex=True,
                            sharey=True,
                            squeeze=False)
    labels = list(groups)
    for w, ax in enumerate(axs[0]):
        for (label, (burden, dies,
                     frames)), colour in zip(groups.items(),
                                             group_colours(labels, colours)):
            in_window = ((frames >= time_edges[w]) &
                         (frames < time_edges[w + 1]))
            b, d = burden[in_window], dies[in_window]
            exposure = np.histogram(b, edges)[0] * hours_per_frame
            deaths = np.histogram(b[d], edges)[0]

            print(f'\n{label_prefix}{label}, '
                  f'{time_edges[w] * hours_per_frame:.0f}-'
                  f'{time_edges[w + 1] * hours_per_frame:.0f} h')
            print(f'{"Burden bin":>24} {"Cell-hours":>11} {"Deaths":>7}')
            for lo, hi, e, d in zip(edges[:-1], edges[1:], exposure, deaths):
                print(f'{lo:>11.0f} - {hi:<10.0f} {e:>11.1f} {d:>7d}')

            # Plot each bin at its median burden
            bin_idx = np.digitize(b, edges[1:-1])
            x = np.array([
                np.median(b[bin_idx == i]) if np.any(bin_idx == i) else np.nan
                for i in range(len(edges) - 1)
            ])
            print(label, b)
            with np.errstate(invalid='ignore', divide='ignore'):
                hazard = deaths / exposure
                err = np.sqrt(deaths) / exposure
            ax.errorbar(x,
                        hazard,
                        yerr=err,
                        color=colour,
                        marker='o',
                        markersize=3,
                        capsize=2,
                        label=f'{label_prefix}{label}{label_suffix}')
        ax.set_title(
            f'{time_edges[w] * hours_per_frame:.0f}-'
            f'{time_edges[w + 1] * hours_per_frame:.0f} h',
            fontsize=9)
        ax.set_xscale('symlog', linthresh=edges[2])
        # ax.set_yscale('log')
        ax.set_xlabel(f'Fluor Total {lag * hours_per_frame} hours earlier')
        ax.grid()
    axs[0, 0].set_ylabel('Deaths / cell-hour at risk')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    plt.savefig(save_as)
