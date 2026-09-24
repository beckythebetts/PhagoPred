"""Retro-fit boundary smoothing onto datasets segmented before the upsampling fix.

Segmentations produced by the old ``cv2.INTER_NEAREST`` path in
:mod:`~PhagoPred.cellpose_segmentation.segment` carry a staircase of ~``k``
pixel steps on every contour (see :mod:`upsample` for the full explanation).
This rewrites ``Segmentations/<channel>`` in place with the staircase removed.

    python -m PhagoPred.cellpose_segmentation.resmooth PhagoPred/Datasets/E.h5

The pass is resumable and refuses to smooth a dataset twice: it records
``SmoothingSigmaLowRes`` and ``SmoothedFrames`` on the ``Segmentations`` group.
Cell identities are preserved, so tracking stays valid, but **any features
already extracted from these masks are invalidated and must be recomputed**.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.cellpose_segmentation.upsample import (DEFAULT_SIGMA_LOW_RES,
                                                      smooth_label_image)


def scale_factor_from_model(model_dir: Path) -> float:
    """Read ``diam_labels / diam_mean`` straight out of a cellpose checkpoint."""
    import torch

    model_dir = Path(model_dir)
    checkpoint = model_dir / 'models' / 'model'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'no cellpose checkpoint at {checkpoint}')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    return float(state['diam_labels'].item() / state['diam_mean'].item())


def resmooth_dataset(h5_file: Path,
                     channel: str = 'Phase',
                     sigma_low_res: float = DEFAULT_SIGMA_LOW_RES,
                     scale_factor: float = None,
                     force: bool = False) -> None:
    """Smooth every frame of ``Segmentations/<channel>`` in place.

    Args:
        h5_file: dataset to rewrite.
        channel: segmentation channel to process.
        sigma_low_res: bandwidth in low-resolution pixels; 0.5 is half the
            quantisation step.
        scale_factor: the factor the masks were enlarged by.  Read from the
            model recorded on the ``Segmentations`` group if not given.
        force: re-run even if the dataset is already marked as smoothed.
    """
    h5_file = Path(h5_file)

    with h5py.File(h5_file, 'r+') as f:
        group = f['Segmentations']
        segmentations = group[channel]

        done = int(group.attrs.get('SmoothedFrames', 0))
        n_frames = segmentations.shape[0]
        if done >= n_frames and not force:
            print(f'{h5_file.name}/{channel}: already smoothed '
                  f'(sigma={group.attrs.get("SmoothingSigmaLowRes")}), skipping')
            return
        if force:
            done = 0

        if scale_factor is None:
            model_dir = group.attrs.get('Model', str(SETTINGS.CELLPOSE_MODEL))
            scale_factor = scale_factor_from_model(Path(model_dir))

        sigma_full = sigma_low_res * scale_factor
        print(f'{h5_file.name}/{channel}: k={scale_factor:.3f}, '
              f'sigma={sigma_full:.2f} full-res px, '
              f'frames {done}-{n_frames}')

        group.attrs['SmoothingSigmaLowRes'] = sigma_low_res
        group.attrs['SmoothingScaleFactor'] = scale_factor

        for frame_idx in tqdm(range(done, n_frames), desc='Smoothing'):
            frame = segmentations[frame_idx]
            if (frame >= 0).any():
                frame = smooth_label_image(frame,
                                           scale_factor=scale_factor,
                                           background=-1,
                                           sigma_low_res=sigma_low_res)
                segmentations[frame_idx] = frame.astype(segmentations.dtype)
            # Checkpoint after every frame so an interrupted run resumes here.
            group.attrs['SmoothedFrames'] = frame_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('h5_files', nargs='+', type=Path)
    parser.add_argument('--channel', default='Phase')
    parser.add_argument('--sigma',
                        type=float,
                        default=DEFAULT_SIGMA_LOW_RES,
                        help='bandwidth in low-resolution pixels (default 0.5)')
    parser.add_argument('--scale-factor',
                        type=float,
                        default=None,
                        help='override k instead of reading it from the model')
    parser.add_argument('--force',
                        action='store_true',
                        help='re-smooth a dataset that is already marked done')
    args = parser.parse_args()

    for h5_file in args.h5_files:
        resmooth_dataset(h5_file,
                         channel=args.channel,
                         sigma_low_res=args.sigma,
                         scale_factor=args.scale_factor,
                         force=args.force)


if __name__ == '__main__':
    main()
