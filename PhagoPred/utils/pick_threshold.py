"""Pick EPI_THRESHOLD by eye: epi frames from several wells, with the pixels
above a slider threshold overlaid and cell outlines for context."""
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
import cv2

from PhagoPred.utils.h5_file import H5File

DATA = Path('~/thor_server/MacrophageData').expanduser()
dirs = (DATA / '02_10', DATA / '16_10')

n_samples = 30

all_files = np.concatenate([list(d.glob('*.h5')) for d in dirs])
print(all_files)

all_frames = np.arange(H5File(all_files[0]).num_frames)

files = np.random.choice(all_files, n_samples)
frames = np.random.choice(all_frames, n_samples)

samples = list(zip(files, frames))

epi, cells = [], []
for well, frame in samples:
    h5 = H5File(well)
    epi.append(
        cv2.GaussianBlur(h5.get_epi(frame), ksize=(0, 0), sigmaX=1,
                         sigmaY=1))  # unthresholded
    cells.append(h5.get_segs(frame) + 1)  # labels need 0 as background
epi, cells = np.stack(epi), np.stack(cells)

viewer = napari.Viewer()
image = viewer.add_image(epi,
                         name='Epi',
                         colormap='gray',
                         contrast_limits=np.percentile(epi, [1, 99.9]))
outlines = viewer.add_labels(cells, name='Cells', opacity=0.4)
outlines.contour = 1
above = viewer.add_labels(np.zeros_like(epi, dtype=np.uint8),
                          name='Above threshold',
                          opacity=0.6)


@magicgui(auto_call=True,
          threshold={
              'widget_type': 'Slider',
              'min': 0,
              'max': 200,
              'step': 5
          })
def set_threshold(threshold: int = H5File.EPI_THRESHOLD):
    above.data = (epi >= threshold).astype(np.uint8)
    viewer.status = f'threshold {threshold}: {above.data.mean():.3%} of pixels'


viewer.window.add_dock_widget(set_threshold, area='right')
set_threshold()
napari.run()
