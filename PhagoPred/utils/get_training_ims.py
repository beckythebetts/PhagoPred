from pathlib import Path

import numpy as np
from .h5_file import H5File
from PIL import Image

dataset_dir = Path('~/thor_server').expanduser() / 'MacrophageData'
save_to = Path(
    '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_and_hicks_20x_clahe_with_rescale/images'
)
num_ims = 5

all_files = []
for direcotry in (dataset_dir / '02_10', dataset_dir / '16_10'):
    for file in direcotry.glob('*.h5'):
        all_files.append(file)

files = np.random.choice(all_files, num_ims, replace=False)

for file in files:
    h5_file = H5File(file)
    num_frames = h5_file.num_frames
    frame = np.random.choice(np.arange(150, num_frames))
    im = h5_file.get_phase(frame)
    Image.fromarray(im).save(save_to /
                             f'{file.parent.name}_{file.stem}_{frame}.png')
