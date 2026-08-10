import sys
from typing import Literal
from cellpose import models, core, plot, io, utils, transforms, dynamics
from cellpose.io import logger_setup
import logging
import cv2
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
        diameter = model.net.diam_labels.item()
        cells_ds = cells_group[category]

        for frame_idx in tqdm(range(images_ds.shape[0]), desc='Segmenting'):
            image = images_ds[frame_idx]
            # resample=False runs the mask dynamics and the flow-error QC at the
            # network's own (rescaled) resolution rather than at full resolution,
            # which is ~3x faster; the masks are upsampled back below.
            masks, _, _ = model.eval(image[:, :, np.newaxis],
                                     diameter=diameter,
                                     batch_size=32,
                                     resample=False)
            masks = masks.astype(np.int16)
            if masks.shape != image.shape:
                masks = cv2.resize(masks, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            masks -= 1

            cell_ids = np.unique(masks)
            cell_ids = cell_ids[cell_ids != -1]
            if cell_ids.size:
                num_instances = int(cell_ids.max()) + 1
                if num_instances > cells_ds.shape[1]:
                    cells_ds.resize(num_instances, axis=1)

            row = np.full(cells_ds.shape[1], np.nan, dtype=np.float32)
            row[cell_ids] = 1
            cells_ds[frame_idx] = row

            segmentations_ds[frame_idx] = masks


def main():
    seg_dataset('/home/ubuntu/PhagoPred/PhagoPred/Datasets/B.h5')
    # segment(SETTINGS.DATASET)


if __name__ == '__main__':
    main()
