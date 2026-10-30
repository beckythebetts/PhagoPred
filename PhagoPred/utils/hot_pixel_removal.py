from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

import matplotlib.pyplot as plt

from .h5_file import H5File

HOT_PIXEL_PATH = Path('PhagoPred') / 'utils' / 'hot_pixel_mask.npy'
RESIDUAL_THRESHOLD = 100


def _get_residuals(h5_files: list[Path], n_frames=5) -> np.ndarray:
    residuals = []
    with tqdm(total=n_frames * len(h5_files)) as pbar:
        for h5_file in h5_files:
            file = H5File(h5_file)
            frames = np.random.choice(np.arange(0, file.num_frames, dtype=int),
                                      n_frames,
                                      replace=False)
            epi = []
            for frame in frames:
                epi.append(file.get_epi(frame))
                pbar.update()
            epi = np.stack(epi).astype(np.float32)
            temporal_median = np.median(epi, axis=0)

            residuals.append(temporal_median -
                             median_filter(temporal_median, size=3))

    return np.stack(residuals, axis=0)


def estimate_residual_threshold(
        h5_files: list[Path],
        n_frames: int = 5,
        save_as: Path = Path('temp') / 'residual_threshold.png') -> None:
    """Plot, per threshold, the fraction of pixels flagged in any file that
    are flagged in all of them, and how many pixels that flags."""
    residuals = _get_residuals(h5_files, n_frames)
    thresholds = np.arange(5, 1000, 5)

    agreement, num_flagged = [], []
    for t in tqdm(thresholds, desc='Testing thresholds'):
        flagged = residuals > t
        in_all = np.all(flagged, axis=0).sum()
        in_any = np.any(flagged, axis=0).sum()
        agreement.append(in_all / max(in_any, 1))
        num_flagged.append(in_all)

    fig, ax = plt.subplots()
    ax.plot(thresholds, agreement)
    ax.set_xlabel('Residual threshold')
    ax.set_ylabel('Flagged in all files / flagged in any')
    ax2 = ax.twinx()
    ax2.semilogy(thresholds, num_flagged, color='C1')
    ax2.set_ylabel('Pixels flagged in all files', color='C1')
    fig.tight_layout()
    fig.savefig(save_as)


def estimate_hot_pixel_mask(h5_files: Path,
                            n_frames=5,
                            save_as: Path = HOT_PIXEL_PATH) -> None:
    mask = np.all(_get_residuals(h5_files, n_frames) > RESIDUAL_THRESHOLD,
                  axis=0)
    np.save(save_as, mask)


def get_mask(path: Path = HOT_PIXEL_PATH) -> np.ndarray:
    return np.load(path)


def correct_hot_pixels(im: np.ndarray,
                       mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        mask = get_mask()

    ys, xs = np.nonzero(mask)
    padded = np.pad(im, 1, mode='reflect')
    neighbourhoods = np.stack(
        [padded[ys + dy, xs + dx] for dy in range(3) for dx in range(3)])
    im[ys, xs] = np.median(neighbourhoods, axis=0).astype(im.dtype)
    return im


if __name__ == '__main__':
    # estimate_residual_threshold(
    #     list((Path('~/thor_server').expanduser() / 'MacrophageData' /
    #           '16_10').glob('*.h5')))
    estimate_hot_pixel_mask(
        list((Path('~/thor_server').expanduser() / 'MacrophageData' /
              '16_10').glob('*.h5')))
