"""Fluorescence per cell over time, per group."""
from __future__ import annotations
from functools import partial

from pathlib import Path

import numpy as np

from PhagoPred.utils.h5_file import H5File
from ._common import plot_by_label, frame_time, alive_and_deaths, alive_mask


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
    plot_by_label(label_to_files,
                  partial(_well_fluor,
                          per_cell=per_cell,
                          cell_number=cell_number),
                  ylabel='Total fluorescence per cell',
                  save_as=save_as,
                  label_prefix=label_prefix,
                  label_suffix=label_suffix,
                  colours=colours,
                  log_y=True,
                  percentile=0.05,
                  **kwargs)
