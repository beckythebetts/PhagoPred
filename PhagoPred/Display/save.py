import h5py
import torch
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_tracked_images(dir, first_frame=0, last_frame=50):
    print('\nPREPARING TRACKED IMAGES\n')
    hdf5_file = SETTINGS.DATASET
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = tools.get_images(f, 'Phase', first_frame, last_frame)
        segmentation_data = tools.get_masks(f, 'Phase', first_frame, last_frame)
    max_cell_index = np.max(segmentation_data)
    LUT = torch.randint(low=10, high=255, size=(max_cell_index+1, 3)).to(device)
    LUT[0] = torch.tensor([0, 0, 0]).to(device)
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(rgb_phase, segmentation_data)):
        segmentation = torch.tensor(segmentation).to(device)
        phase_image = torch.tensor(phase_image).to(device).int()
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        outlines = mask_funcs.mask_outlines(segmentation, thickness=1).int()
        outlines = LUT[outlines]
        phase_image =  (torch.where(outlines>0, outlines, phase_image))
        phase_image = phase_image.cpu().numpy().astype(np.uint8)
        tracked[i] = phase_image
    for i, im in enumerate(tracked):
        cv2.putText(im, text=f'frame: {i+first_frame} | time: {(i+first_frame)*SETTINGS.TIME_STEP}/s', org=(20, SETTINGS.IMAGE_SIZE[1]-20), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale=3, color=(255, 255, 255), thickness=3)
        plt.imsave(dir / f'{i}.png', im/256)
    
if __name__ == '__main__':
    show_tracked_images(Path(r'C:\Users\php23rjb\Downloads\temp') / 'test_track_del', last_frame=10)