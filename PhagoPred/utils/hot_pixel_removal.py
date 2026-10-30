from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

import matplotlib.pyplot as plt

from .h5_file import H5File

HOT_PIXEL_PATH = Path('PhagoPred') / 'utils' / 'hot_pixel_mask.npy'
RESIDUAL_THRESHOLD = 10


def _get_residual(h5_files: Path, n_frames=5) -> np.ndarray:
    epi = []
    with tqdm(total=n_frames * len(h5_files)) as pbar:
        for h5_file in h5_files:
            file = H5File(h5_file)
            frames = np.random.choice(np.arange(0, file.num_frames, dtype=int),
                                      n_frames)
            for frame in frames:
                epi.append(file.get_epi(frame))
                pbar.update()
        epi = np.stack(epi).astype(np.float32)
        temporal_median = np.median(epi, axis=0)

    residual = temporal_median - median_filter(temporal_median, size=3)

    return residual


def estimate_residual_threshold(
        h5_files: list[Path],
        n_frames=5,
        save_hist_as: Path = Path('temp') / 'residuals_hist.png') -> None:
    residual = _get_residual(h5_files, n_frames)

    plt.hist(residual.flatten(), bins=500, log=True)
    plt.savefig(save_hist_as)


def estimate_hot_pixel_mask(h5_files: Path,
                            n_frames=5,
                            save_as: Path = HOT_PIXEL_PATH) -> None:
    residual = _get_residual(h5_files, n_frames)
    mask = residual > RESIDUAL_THRESHOLD
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
    estimate_residual_threshold(
        list((Path('~/thor_server').expanduser() / 'MacrophageData' /
              '16_10').glob('*.h5')))
