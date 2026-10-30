"""Fluorescence per cell over time, per group."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from PhagoPred.utils.h5_file import H5File
from ._common import plot_by_label, frame_time


def _well_tot_fluor(h5_file: Path,
                    stride: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Summed Epi intensity at each frame of one well."""
    h5 = H5File(h5_file)
    tot_fluor = h5.total_fluorescence
    return np.arange(len(tot_fluor)) * frame_time(h5_file), tot_fluor


def plot_tot_fluor(
    label_to_files: dict[str | int, list[Path]],
    save_as: Path = Path('temp') / 'total_fluor.png',
    label_prefix: str = 'MOI ',
    label_suffix: str = "",
    colours: dict | None = None,
    **kwargs,
) -> None:
    """Total fluorescence per well over time; kwargs go to plot_by_label."""
    plot_by_label(label_to_files,
                  _well_tot_fluor,
                  ylabel='Total fluorescence per well',
                  save_as=save_as,
                  label_prefix=label_prefix,
                  label_suffix=label_suffix,
                  colours=colours,
                  **kwargs)
