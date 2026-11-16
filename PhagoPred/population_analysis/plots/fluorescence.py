"""Fluorescence per cell over time, per group."""
from __future__ import annotations
from functools import partial

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.h5_file import H5File
from ._common import plot_by_label, frame_time, alive_and_deaths, alive_mask, group_colours


def _well_fluor(h5_file: Path,
                per_cell: bool = True,
                cell_number: bool = False,
                infected_thresh: float = 500,
                before_event: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Summed Epi intensity at each frame of one well."""
    h5 = H5File(h5_file)
    frames = np.arange(h5.num_frames)
    counted = alive_mask(h5_file, frames)
    if before_event:
        # NaN (no event) compares False, so those cells are kept throughout
        counted &= ~(h5.event_frames <= frames[:, None])

    fluor = h5.get_features('Fluor Total')
    values = fluor > infected_thresh if cell_number else fluor
    values = np.where(counted, values, 0).sum(axis=h5.CELLS_DIM)
    if per_cell:
        with np.errstate(invalid='ignore', divide='ignore'):
            values = values / counted.sum(axis=h5.CELLS_DIM)
    return frames * frame_time(h5_file), values


# fluorescence.py
def _cell_fluor(h5_file: Path, before_event: bool = True):
    """Per-cell Fluor Total, NaN where not alive / after event. (frames, cells)"""
    h5 = H5File(h5_file)
    frames = np.arange(h5.num_frames)
    counted = alive_mask(h5_file, frames)
    if before_event:
        counted &= ~(h5.event_frames <= frames[:, None])
    fluor = h5.get_features('Fluor Total').astype(float)
    return frames * frame_time(h5_file), np.where(counted, fluor, np.nan)


def plot_tot_fluor(
    label_to_files: dict[str | int, list[Path]],
    save_as: Path = Path('temp') / 'total_fluor.png',
    label_prefix: str = 'MOI ',
    label_suffix: str = "",
    colours: dict | None = None,
    per_cell: bool = True,
    cell_number: bool = False,
    **kwargs,
) -> None:
    """Total fluorescence per well over time; kwargs go to plot_by_label."""
    plot_by_label(
        label_to_files,
        partial(_cell_fluor,
                #   per_cell=per_cell,
                #   cell_number=cell_number,
                ),
        ylabel='Total fluorescence per cell',
        save_as=save_as,
        label_prefix=label_prefix,
        label_suffix=label_suffix,
        colours=colours,
        log_y=True,
        percentile=5,
        pool_cells=True,
        **kwargs)


def plot_fluor_hist(label_to_files,
                    save_as=Path('temp') / 'fluor_hist.png',
                    label_prefix='MOI ',
                    colours=None,
                    control=0,
                    control_pct=99,
                    n_bins=80,
                    per_cell_max=False):
    labels = sorted(label_to_files, key=float)
    colours = group_colours(labels, colours)
    pooled = {}
    for label in labels:
        vals = []
        for f in label_to_files[label]:
            _, v = _cell_fluor(f)  # (frames, cells)
            v = np.nanmax(v, axis=0) if per_cell_max else v.ravel()
            vals.append(v[~np.isnan(v)])
        pooled[label] = np.concatenate(vals)

    pos = np.concatenate([v[v > 0] for v in pooled.values()])
    bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()), n_bins)

    fig, ax = plt.subplots()
    for c, label in zip(colours, labels):
        v = pooled[label]
        pos = v[v > 0]
        ax.hist(
            pos,
            bins=bins,
            weights=np.full(len(pos), 1 / len(v)),
            histtype='step',
            color=c,
            label=
            f'{label_prefix}{label} (n={len(v)}, zero: {(v <= 0).mean():.0%})')
    if control in pooled:
        thresh = np.percentile(pooled[control], control_pct)
        ax.axvline(thresh,
                   color='k',
                   ls='--',
                   label=f'{control_pct}th pct of control = {thresh:.0f}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Fluor Total')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_as)
