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
    missed: dict[int, int] | None = None,
    true: dict[int, int] | None = None,
    show_counts: bool = False,
    save_as: Path = Path('temp') / 'missed_cells.png',
) -> None:
    """Proportion of cells the segmentation missed over time, and optionally
    the automated and true (automated + missed) counts above it.

    Args:
        missed: {frame: cells missed}, counted with the GUI's counter.
    """
    frames = np.array(sorted(missed)) if missed is not None else np.array(
        sorted(true))
    auto = automated_counts(h5_file, frames)
    if missed is not None:
        num_missed = np.array([missed[f] for f in frames])
        true = auto + num_missed
    elif true is not None:
        true = np.array([true[f] for f in frames])
        num_missed = None

    if num_missed is None:
        num_missed = true - auto
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
        '~/thor_server').expanduser() / 'MacrophageData' / '06_11' / 'B4.h5'
    # h5_file = Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/A1.h5')
    true_counts_1 = {
        0: 171,
        20: 157,
        40: 154,
        60: 154,
        80: 154,
        100: 150,
        120: 147,
        140: 127,
        160: 118,
        180: 107,
        200: 109,
        220: 104,
        240: 97,
        260: 88,
        280: 82
    }  # first count (against the old 40/60 px tracking)
    true_counts_2 = {
        0: 175,
        20: 166,
        40: 160,
        60: 157,
        80: 159,
        100: 156,
        120: 145,
        140: 143,
        160: 125,
        180: 116,
        200: 117,
        220: 106,
        240: 99,
        260: 84,
        280: 77
    }  # second count (against the new tracking)
    true_counts = {
        t: (true_counts_1[t] + true_counts_2[t]) / 2
        for t in true_counts_1
    }
    b4_missed = {
        0: 8,
        40: 4,
        80: 8,
        120: 6,
        160: 1,
        200: 5,
        240: 5,
        280: 7,
    }

    plot_missed_cells(h5_file, missed=b4_missed, show_counts=True)
