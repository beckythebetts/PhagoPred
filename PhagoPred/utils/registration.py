from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from PhagoPred.utils.h5_file import H5File


def estimate_frame_offsets(h5_file: Path, downsample: int = 2) -> np.ndarray:
    """(T, 2) integer (row, col) displacement of each frame's content from
    the median position, chaining phase correlation between consecutive
    Phase frames. Relative to the median rather than frame 0, as frame 0 is
    itself often displaced, so typical frames barely move.
    """
    file = H5File(h5_file)
    H, W = file.im_dims
    size = (W // downsample, H // downsample)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    steps = np.zeros((file.num_frames, 2))
    prev = None
    for t in tqdm(range(file.num_frames), desc='Estimating frame shifts'):
        im = cv2.resize(file.get_phase(t), size,
                        interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(prev, im, window)
            steps[t] = (dy * downsample, dx * downsample)
        prev = im
    # Round the running total, not each step, so rounding errors don't add up
    offsets = np.cumsum(steps, axis=0)
    return np.round(offsets - np.median(offsets, axis=0)).astype(int)


def shift_frame(im: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """im moved back by offset so it lines up with the median position; the
    uncovered edge is filled with the frame's median."""
    dr, dc = (int(o) for o in offset)
    H, W = im.shape
    out = np.full_like(im, np.median(im[::8, ::8]))
    out[max(0, -dr):H - max(0, dr), max(0, -dc):W - max(0, dc)] = \
        im[max(0, dr):H - max(0, -dr), max(0, dc):W - max(0, -dc)]
    return out


def register_frames(h5_file: Path, downsample: int = 2) -> np.ndarray:
    """Align every frame to the median position, applying the same shifts to
    Phase and Epi, and record the offsets."""
    file = H5File(h5_file)
    offsets = estimate_frame_offsets(h5_file, downsample)
    file.write_phase(
        (shift_frame(file.get_phase(t), o) for t, o in enumerate(offsets)),
        overwrite=True)
    if file.ims_info(H5File.FLUOR) is not None:
        file.write_epi(
            (shift_frame(file.get_epi(t), o) for t, o in enumerate(offsets)),
            overwrite=True)
    file.write_attr('Frame offsets (rows, cols)', offsets)
    return offsets
