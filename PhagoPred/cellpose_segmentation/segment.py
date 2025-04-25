import sys
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


class CellposeModel_withsave(models.CellposeModel):
    def eval(self, x, hdf5_file, batch_size=8, resample=True, channels=None, channel_axis=None,
             z_axis=None, normalize=True, invert=False, rescale=None, diameter=None,
             flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False, anisotropy=None,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4, niter=None,
             augment=False, tile=True, tile_overlap=0.1, bsize=224,
             interp=True, compute_masks=True, progress=None):
        """ segment list of images x, or 4D array - Z x nchan x Y x X

        Args:
            x (list, np.ndarry): can be list of 2D/3D/4D images, or array of 2D/3D/4D images
            batch_size (int, optional): number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage). Defaults to 8.
            resample (bool, optional): run dynamics at original image size (will be slower but create more accurate boundaries). Defaults to True.
            channels (list, optional): list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].
                Defaults to None.
            channel_axis (int, optional): channel axis in element of list x, or of np.ndarray x.
                if None, channels dimension is attempted to be automatically determined. Defaults to None.
            z_axis  (int, optional): z axis in element of list x, or of np.ndarray x.
                if None, z dimension is attempted to be automatically determined. Defaults to None.
            normalize (bool, optional): if True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel;
                can also pass dictionary of parameters (all keys are optional, default values shown):
                    - "lowhigh"=None : pass in normalization values for 0.0 and 1.0 as list [low, high] (if not None, all following parameters ignored)
                    - "sharpen"=0 ; sharpen image with high pass filter, recommended to be 1/4-1/8 diameter of cells in pixels
                    - "normalize"=True ; run normalization (if False, all following parameters ignored)
                    - "percentile"=None : pass in percentiles to use as list [perc_low, perc_high]
                    - "tile_norm"=0 ; compute normalization in tiles across image to brighten dark areas, to turn on set to window size in pixels (e.g. 100)
                    - "norm3D"=False ; compute normalization across entire z-stack rather than plane-by-plane in stitching mode.
                Defaults to True.
            invert (bool, optional): invert image pixel intensity before running network. Defaults to False.
            rescale (float, optional): resize factor for each image, if None, set to 1.0;
                (only used if diameter is None). Defaults to None.
            diameter (float, optional):  diameter for each image,
                if diameter is None, set to diam_mean or diam_train if available. Defaults to None.
            flow_threshold (float, optional): flow error threshold (all cells with errors below threshold are kept) (not used for 3D). Defaults to 0.4.
            cellprob_threshold (float, optional): all pixels with value above threshold kept for masks, decrease to find more and larger masks. Defaults to 0.0.
            do_3D (bool, optional): set to True to run 3D segmentation on 3D/4D image input. Defaults to False.
            anisotropy (float, optional): for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y). Defaults to None.
            stitch_threshold (float, optional): if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Defaults to 0.0.
            min_size (int, optional): all ROIs below this size, in pixels, will be discarded. Defaults to 15.
            max_size_fraction (float, optional): max_size_fraction (float, optional): Masks larger than max_size_fraction of
                total image size are removed. Default is 0.4.
            niter (int, optional): number of iterations for dynamics computation. if None, it is set proportional to the diameter. Defaults to None.
            augment (bool, optional): tiles image with overlapping tiles and flips overlapped regions to augment. Defaults to False.
            tile (bool, optional): tiles image to ensure GPU/CPU memory usage limited (recommended). Defaults to True.
            tile_overlap (float, optional): fraction of overlap of tiles when computing flows. Defaults to 0.1.
            bsize (int, optional): block size for tiles, recommended to keep at 224, like in training. Defaults to 224.
            interp (bool, optional): interpolate during 2D dynamics (not available in 3D) . Defaults to True.
            compute_masks (bool, optional): Whether or not to compute dynamics and return masks. This is set to False when retrieving the styles for the size model. Defaults to True.
            progress (QProgressBar, optional): pyqt progress bar. Defaults to None.

        Returns:
            A tuple containing:
                - masks (list, np.ndarray): labelled image(s), where 0=no masks; 1,2,...=mask labels
                - flows (list): list of lists: flows[k][0] = XY flow in HSV 0-255; flows[k][1] = XY(Z) flows at each pixel; flows[k][2] = cell probability (if > cellprob_threshold, pixel used for dynamics); flows[k][3] = final pixel locations after Euler integration
                - styles (list, np.ndarray): style vector summarizing each image of size 256.

        """

        models_logger = logging.getLogger(__name__)
        if isinstance(x, list) or x.squeeze().ndim == 5:

            self.timing = []
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out,
                              mininterval=30) if nimg > 1 else range(nimg)
            for i in iterator:
                tic = time.time()
                maski, flowi, stylei = self.eval(
                    x[i], hdf5_file, batch_size=batch_size,
                    channels=channels[i] if channels is not None and
                                            ((len(channels) == len(x) and
                                              (isinstance(channels[i], list) or
                                               isinstance(channels[i], np.ndarray)) and len(channels[i]) == 2))
                    else channels, channel_axis=channel_axis, z_axis=z_axis,
                    normalize=normalize, invert=invert,
                    rescale=rescale[i] if isinstance(rescale, list) or
                                          isinstance(rescale, np.ndarray) else rescale,
                    diameter=diameter[i] if isinstance(diameter, list) or
                                            isinstance(diameter, np.ndarray) else diameter, do_3D=do_3D,
                    anisotropy=anisotropy, augment=augment, tile=tile,
                    tile_overlap=tile_overlap, bsize=bsize, resample=resample,
                    interp=interp, flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold, compute_masks=compute_masks,
                    min_size=min_size, max_size_fraction=max_size_fraction,
                    stitch_threshold=stitch_threshold,
                    progress=progress, niter=niter)
                with h5py.File(hdf5_file, 'r+') as f:
                    # If any detected instances are > 75% epi, remove them from the mask:
                    epi_im = f['Segmentations']['Epi'][f'{int(i):04}'][:]
                    overlap_idxs, counts = np.unique(maski[np.logical_and(maski>0, epi_im>0)], return_counts=True)
                    for idx, count in zip(overlap_idxs, counts):
                        if count / np.count_nonzero(maski[maski==idx]) > 0.4 and idx !=0:
                            maski[maski==idx] = 0
                    f.create_dataset(f'Segmentations/Phase/{int(i):04}', dtype='i2', data=maski)
                self.timing.append(time.time() - tic)
            return masks, flows, styles
        else:
            # reshape image
            x = transforms.convert_image(x, channels, channel_axis=channel_axis,
                                         z_axis=z_axis, do_3D=(do_3D or
                                                               stitch_threshold > 0),
                                         nchan=self.nchan)
            if x.ndim < 4:
                x = x[np.newaxis, ...]

            if diameter is not None and diameter > 0:
                rescale = self.diam_mean / diameter
            elif rescale is None:
                diameter = self.diam_labels
                rescale = self.diam_mean / diameter

            masks, styles, dP, cellprob, p = self._run_cp(
                x, compute_masks=compute_masks, normalize=normalize, invert=invert,
                rescale=rescale, resample=resample, augment=augment, tile=tile,
                batch_size=batch_size, tile_overlap=tile_overlap, bsize=bsize, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, interp=interp, min_size=min_size,
                max_size_fraction=max_size_fraction, do_3D=do_3D, anisotropy=anisotropy, niter=niter,
                stitch_threshold=stitch_threshold)

            flows = [plot.dx_to_circ(dP), dP, cellprob, p]
            return masks, flows, styles


def segment(hdf5_file):
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d' % use_GPU)
    logger_setup()

    model = CellposeModel_withsave(gpu=use_GPU, pretrained_model=str(SETTINGS.CELLPOSE_MODEL / 'models' / 'model'))
    channels = [0, 0]
    with h5py.File(hdf5_file, 'r+') as f:
        if 'Segmentations' in f:
            del f['Segmentations']
        ims = [f['Images']['Phase'][frame][:] for frame in f['Images']['Phase'].keys()]
    threshold_epi.main()
    model.eval(ims, hdf5_file, diameter=None, flow_threshold=0.2, channels=channels)
    with h5py.File(hdf5_file, 'r+') as f:
        f['Segmentations']['Phase'].attrs['Model'] = str(SETTINGS.CELLPOSE_MODEL)


def main():
    segment(SETTINGS.DATASET)


if __name__ == '__main__':
    main()