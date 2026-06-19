import sys
from typing import Literal
from cellpose import models, core, plot, io, utils, transforms, dynamics
from cellpose.io import logger_setup
import logging
import h5py
import numpy as np
import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch

from PhagoPred import SETTINGS
from PhagoPred.cellpose_segmentation import threshold_epi

# class CellposeModel_withsave(models.CellposeModel):

#     def eval(self,
#              x,
#              hdf5_file,
#              batch_size=8,
#              resample=True,
#              channels=None,
#              channel_axis=None,
#              z_axis=None,
#              normalize=True,
#              invert=False,
#              rescale=None,
#              diameter=None,
#              flow_threshold=0.4,
#              cellprob_threshold=0.0,
#              do_3D=False,
#              anisotropy=None,
#              stitch_threshold=0.0,
#              min_size=15,
#              max_size_fraction=0.4,
#              niter=None,
#              augment=False,
#              tile=True,
#              tile_overlap=0.1,
#              bsize=224,
#              interp=True,
#              compute_masks=True,
#              progress=None):
#         """
#         """

#         models_logger = logging.getLogger(__name__)
#         if isinstance(x, list) or x.squeeze().ndim == 5:

#             self.timing = []
#             masks, styles, flows = [], [], []
#             tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
#             nimg = len(x)
#             iterator = trange(nimg, file=tqdm_out,
#                               mininterval=30) if nimg > 1 else range(nimg)
#             for i in iterator:
#                 tic = time.time()
#                 maski, flowi, stylei = self.eval(
#                     x[i],
#                     hdf5_file,
#                     batch_size=batch_size,
#                     channels=channels[i] if channels is not None and
#                     ((len(channels) == len(x) and
#                       (isinstance(channels[i], list) or isinstance(
#                           channels[i], np.ndarray)) and len(channels[i]) == 2))
#                     else channels,
#                     channel_axis=channel_axis,
#                     z_axis=z_axis,
#                     normalize=normalize,
#                     invert=invert,
#                     rescale=rescale[i] if isinstance(rescale, list)
#                     or isinstance(rescale, np.ndarray) else rescale,
#                     diameter=diameter[i] if isinstance(diameter, list)
#                     or isinstance(diameter, np.ndarray) else diameter,
#                     do_3D=do_3D,
#                     anisotropy=anisotropy,
#                     augment=augment,
#                     tile=tile,
#                     tile_overlap=tile_overlap,
#                     bsize=bsize,
#                     resample=resample,
#                     interp=interp,
#                     flow_threshold=flow_threshold,
#                     cellprob_threshold=cellprob_threshold,
#                     compute_masks=compute_masks,
#                     min_size=min_size,
#                     max_size_fraction=max_size_fraction,
#                     stitch_threshold=stitch_threshold,
#                     progress=progress,
#                     niter=niter)
#                 with h5py.File(hdf5_file, 'r+') as f:
#                     # If any detected instances are > 75% epi, remove them from the mask:
#                     epi_im = f['Segmentations']['Epi'][f'{int(i):04}'][:]
#                     overlap_idxs, counts = np.unique(maski[np.logical_and(
#                         maski > 0, epi_im > 0)],
#                                                      return_counts=True)
#                     for idx, count in zip(overlap_idxs, counts):
#                         if count / np.count_nonzero(
#                                 maski[maski == idx]) > 0.4 and idx != 0:
#                             maski[maski == idx] = 0
#                     f.create_dataset(f'Segmentations/Phase/{int(i):04}',
#                                      dtype='i2',
#                                      data=maski)
#                 self.timing.append(time.time() - tic)
#             return masks, flows, styles
#         else:
#             # reshape image
#             x = transforms.convert_image(x,
#                                          channels,
#                                          channel_axis=channel_axis,
#                                          z_axis=z_axis,
#                                          do_3D=(do_3D or stitch_threshold > 0),
#                                          nchan=self.nchan)
#             if x.ndim < 4:
#                 x = x[np.newaxis, ...]

#             if diameter is not None and diameter > 0:
#                 rescale = self.diam_mean / diameter
#             elif rescale is None:
#                 diameter = self.diam_labels
#                 rescale = self.diam_mean / diameter

#             masks, styles, dP, cellprob, p = self._run_cp(
#                 x,
#                 compute_masks=compute_masks,
#                 normalize=normalize,
#                 invert=invert,
#                 rescale=rescale,
#                 resample=resample,
#                 augment=augment,
#                 tile=tile,
#                 batch_size=batch_size,
#                 tile_overlap=tile_overlap,
#                 bsize=bsize,
#                 flow_threshold=flow_threshold,
#                 cellprob_threshold=cellprob_threshold,
#                 interp=interp,
#                 min_size=min_size,
#                 max_size_fraction=max_size_fraction,
#                 do_3D=do_3D,
#                 anisotropy=anisotropy,
#                 niter=niter,
#                 stitch_threshold=stitch_threshold)

#             flows = [plot.dx_to_circ(dP), dP, cellprob, p]
#             return masks, flows, styles

# def segment(hdf5_file):
#     use_GPU = core.use_gpu()
#     print('>>> GPU activated? %d' % use_GPU)
#     logger_setup()

#     model = CellposeModel_withsave(
#         gpu=use_GPU,
#         pretrained_model=str(SETTINGS.CELLPOSE_MODEL / 'models' / 'model'))
#     channels = [0, 0]
#     with h5py.File(hdf5_file, 'r+') as f:
#         if 'Segmentations' in f:
#             del f['Segmentations']
#         ims = [
#             f['Images']['Phase'][frame][:]
#             for frame in f['Images']['Phase'].keys()
#         ]
#     # threshold_epi.main()
#     model.eval(ims,
#                hdf5_file,
#                diameter=None,
#                flow_threshold=0.2,
#                channels=channels)
#     with h5py.File(hdf5_file, 'r+') as f:
#         f['Segmentations']['Phase'].attrs['Model'] = str(
#             SETTINGS.CELLPOSE_MODEL)


def seg_dataset(h5_file: Path,
                model_dir: Path = SETTINGS.CELLPOSE_MODEL,
                channel: Literal['Phase', 'Epi'] = 'Phase',
                category: str = 'Macrophage') -> None:

    model = models.CellposeModel(gpu=True,
                                 pretrained_model=str(model_dir / 'models' /
                                                      'model'))
    with h5py.File(h5_file, 'r+') as f:

        for group_name in ('Segmentations', 'Cells'):
            group = f.require_group(group_name)
            if channel in group:
                del group[channel]
        f['Segmentations'].attrs['Model'] = str(model_dir)
        images_ds = f['Images'][channel]
        segmentations_ds = f.create_dataset(f'Segmentations/{channel}',
                                            shape=images_ds.shape,
                                            maxshape=images_ds.shape,
                                            dtype='i2')
        cells_group = f.require_group(f'Cells/{channel}')
        cells_group.require_dataset('Confidence Score',
                                    shape=(images_ds.shape[0], 0),
                                    maxshape=(images_ds.shape[0], None),
                                    dtype=np.float32,
                                    exact=True,
                                    fillvalue=np.nan)

        cells_group.require_dataset(category,
                                    shape=(images_ds.shape[0], 0),
                                    maxshape=(images_ds.shape[0], None),
                                    dtype=np.float32,
                                    exact=True,
                                    fillvalue=np.nan)
        for frame_idx in tqdm(range(images_ds.shape[0]), desc='Segmenting'):
            image = images_ds[frame_idx]
            masks, _, _ = model.eval(image[:, :, np.newaxis], batch_size=64)
            masks = masks.astype(np.int16)
            masks -= 1

            num_instances = len(np.unique(masks)) - 1

            current_max_instances = cells_group[category].shape[1] - 1
            if num_instances > current_max_instances:
                cells_group[category].resize(num_instances, axis=1)

            for i in np.unique(masks):
                if i != -1:
                    cells_group[category][frame_idx, i] = 1

            segmentations_ds[frame_idx] = masks


def main():
    seg_dataset('/home/ubuntu/PhagoPred/PhagoPred/Datasets/B.h5')
    # segment(SETTINGS.DATASET)


if __name__ == '__main__':
    main()
