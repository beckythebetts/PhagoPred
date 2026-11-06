"""Segmentation misses for one well, against cells counted by hand with the
GUI's missed cell counter."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.h5_file import H5File
from ._common import frame_time


def automated_counts(h5_file: Path, frames: np.ndarray) -> np.ndarray:
    """Cells in the GUI's Segmentations layer (good tracks, alive) at each of
    frames."""
    h5 = H5File(h5_file)
    counts = []
    for frame in frames:
        segs = h5.get_segs(int(frame), filter_bad=True, alive_only=True)
        counts.append(len(np.unique(segs[segs >= 0])))
    return np.array(counts)


def plot_missed_cells(
    h5_file: Path,
    missed: dict[int, int],
    show_counts: bool = False,
    save_as: Path = Path('temp') / 'missed_cells.png',
) -> None:
    """Proportion of cells the segmentation missed over time, and optionally
    the automated and true (automated + missed) counts above it.

    Args:
        missed: {frame: cells missed}, counted with the GUI's counter.
    """
    frames = np.array(sorted(missed))
    num_missed = np.array([missed[f] for f in frames])
    auto = automated_counts(h5_file, frames)
    true = auto + num_missed
    times = frames * frame_time(h5_file)

    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(2 if show_counts else 1,
                            1,
                            sharex=True,
                            squeeze=False,
                            figsize=(6, 6 if show_counts else 3.5))
    axs = axs[:, 0]
    if show_counts:
        axs[0].plot(times, true, 'o-', label='True (automated + missed)')
        axs[0].plot(times, auto, 'o-', label='Automated')
        axs[0].set_ylabel('Cells')
        axs[0].legend()
    axs[-1].plot(times, num_missed / true, 'o-', color='k')
    axs[-1].set_ylim(0, None)
    axs[-1].set_ylabel('Proportion missed')
    axs[-1].set_xlabel('Time / hours')
    axs[0].set_title(Path(h5_file).stem)

    fig.tight_layout()
    save_as.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_as, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    h5_file = Path(
        '~/thor_server').expanduser() / 'MacrophageData' / '16_10' / 'A1.h5'
    missed = {
        0: 53,
        20: 40,
        40: 41,
        60: 42,
        80: 44,
        100: 42,
        120: 44,
        140: 34,
        160: 32,
        180: 27,
        200: 30,
        220: 30,
        240: 24,
        260: 23,
        280: 20
    }
    plot_missed_cells(h5_file, missed, show_counts=True)
