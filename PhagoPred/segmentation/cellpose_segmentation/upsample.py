from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# Smoothing bandwidth, in units of the low-resolution pixel grid.  0.5 = half
# the quantisation step: it erases sub-grid detail and leaves everything the
# low-res mask could actually resolve.
DEFAULT_SIGMA_LOW_RES = 0.5


def cellpose_scale_factor(model) -> float:
    """Factor by which cellpose shrinks the image, i.e. ``k`` in the docstring.

    Masks produced with ``resample=False`` come back ``k`` times smaller than
    the input image.
    """
    return model.net.diam_labels.item() / model.net.diam_mean.item()


def upsample_labels(labels: np.ndarray,
                    out_shape: tuple[int, int] = None,
                    background: int = 0,
                    smooth_sigma: float = DEFAULT_SIGMA_LOW_RES,
                    margin: int = 4) -> np.ndarray:
    """Enlarge and/or smooth a label image with sub-pixel boundaries.

    Each label is converted to a signed distance field, blurred, resized with
    cubic interpolation and thresholded at 0.  Where two labels both claim a
    pixel the one further inside its own boundary (larger SDF) wins.

    Args:
        labels: 2D integer label image.
        out_shape: ``(rows, cols)`` of the result.  Defaults to the input shape,
            which turns this into a pure boundary smoother -- use that to clean
            up label maps that were already upsampled with ``INTER_NEAREST``,
            passing ``smooth_sigma`` in full-resolution pixels.
        background: label value of the background (cellpose uses 0, our stored
            segmentations use -1).
        smooth_sigma: Gaussian sigma applied to the SDF, in units of the *input*
            grid.  0 disables smoothing.
        margin: background padding (input pixels) kept around each label's
            bounding box so the blurred SDF is not clipped.

    Returns:
        Label image of shape ``out_shape`` and the same dtype as ``labels``.
    """
    if labels.ndim != 2:
        raise ValueError(
            f'expected a 2D label image, got shape {labels.shape}')

    in_shape = labels.shape
    if out_shape is None:
        out_shape = in_shape
    out_shape = (int(out_shape[0]), int(out_shape[1]))

    scale_y = out_shape[0] / in_shape[0]
    scale_x = out_shape[1] / in_shape[1]
    margin = max(margin, int(np.ceil(3 * smooth_sigma)) + 1)

    out = np.full(out_shape, background, dtype=labels.dtype)
    # Depth of the winning label at each output pixel, for resolving overlaps.
    best = np.zeros(out_shape, dtype=np.float32)

    ids = np.unique(labels)
    ids = ids[ids != background]

    for label_id in ids:
        cell = labels == label_id
        coords = np.argwhere(cell)
        r0, c0 = np.maximum(coords.min(axis=0) - margin, 0)
        r1, c1 = np.minimum(coords.max(axis=0) + margin + 1, in_shape)
        crop = cell[r0:r1, c0:c1]

        sdf = (ndimage.distance_transform_edt(crop) -
               ndimage.distance_transform_edt(~crop)).astype(np.float32)
        if smooth_sigma > 0:
            sdf = ndimage.gaussian_filter(sdf, smooth_sigma)

        out_r0, out_c0 = int(round(r0 * scale_y)), int(round(c0 * scale_x))
        out_r1, out_c1 = int(round(r1 * scale_y)), int(round(c1 * scale_x))
        window = (slice(out_r0, out_r1), slice(out_c0, out_c1))

        if (out_r1 - out_r0, out_c1 - out_c0) != crop.shape:
            sdf = cv2.resize(sdf, (out_c1 - out_c0, out_r1 - out_r0),
                             interpolation=cv2.INTER_CUBIC)

        inside = sdf > 0
        if not inside.any():
            # Smoothing dissolved a very small cell; keep its nearest-neighbour
            # footprint rather than losing the detection entirely.
            inside = cv2.resize(crop.astype(np.uint8),
                                (out_c1 - out_c0, out_r1 - out_r0),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            sdf = np.where(inside, np.float32(1e-6), np.float32(-1.0))

        take = inside & (sdf > best[window])
        # `window` is a basic slice, so these write through to `out`/`best`.
        best[window][take] = sdf[take]
        out[window][take] = label_id

    return out


def smooth_label_image(labels: np.ndarray,
                       scale_factor: float,
                       background: int = -1,
                       sigma_low_res: float = DEFAULT_SIGMA_LOW_RES,
                       **kwargs) -> np.ndarray:
    return upsample_labels(labels,
                           out_shape=labels.shape,
                           background=background,
                           smooth_sigma=sigma_low_res * scale_factor,
                           **kwargs)
