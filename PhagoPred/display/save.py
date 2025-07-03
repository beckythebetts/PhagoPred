import h5py
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def save_masks(dir, first_frame=0, last_frame=50):
    hdf5_file = SETTINGS.DATASET
    with h5py.File(hdf5_file, 'r') as f:
        phase_masks = tools.get_masks(f, 'Phase', first_frame, last_frame)
    for i, im in enumerate(phase_masks):
        sys.stdout.write(f'\rFrame {i} / {len(phase_masks)}')
        sys.stdout.flush()
        plt.imsave(dir / f'{i:04}.png', im, cmap='gray')

def save_tracked_images(dir, first_frame=0, last_frame=50):
    """Save the specified frames as .jpegs, with each cell assigned a random colour for visual check of tracking
    """
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print('\nPREPARING TRACKED IMAGES\n')
    os.makedirs(dir, exist_ok=True)
    hdf5_file = SETTINGS.DATASET
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = f['Images']['Phase'][first_frame:last_frame]
        segmentation_data = f['Segmentations']['Phase'][first_frame:last_frame]
    max_cell_index = np.max(segmentation_data)
    LUT = torch.randint(low=10, high=255, size=(max_cell_index+1, 3), dtype=torch.uint8).to(device)

    lut_for_neg1 = torch.tensor([[0, 0, 0]], device=device, dtype=torch.uint8)
    LUT = torch.cat([lut_for_neg1, LUT], dim=0)
    # LUT[0] = torch.tensor([0, 0, 0]).to(device)
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    # tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(rgb_phase, segmentation_data)):
        segmentation = torch.tensor(segmentation).to(device)
        phase_image = torch.tensor(phase_image, dtype=torch.uint8).to(device)
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        outlines = mask_funcs.mask_outlines(segmentation, thickness=3).type(torch.int64)
        colour_outlines = (LUT[outlines]).type(torch.uint8)
        outlined_phase_image =  (torch.where(outlines.unsqueeze(-1).expand_as(colour_outlines)>0, colour_outlines, phase_image))
        outlined_phase_image = outlined_phase_image.cpu().numpy() / 256
        plt.imsave(dir / f'{i}.jpeg', outlined_phase_image)


def overlay_red_on_gray(gray: np.ndarray, red_overlay: np.ndarray,
                                   red_scale: float = 1, normalize: bool = True) -> np.ndarray:
    """
    Overlay a red channel onto a grayscale time series image (3D input).

    Parameters:
        gray (np.ndarray): Grayscale image stack (T, H, W)
        red_overlay (np.ndarray): Red overlay mask (T, H, W), same shape as gray
        red_scale (float): Multiplier for red overlay intensity
        normalize (bool): Whether to normalize both arrays to [0, 1]

    Returns:
        np.ndarray: RGB stack (T, H, W, 3)
    """
    assert gray.shape == red_overlay.shape, "gray and red_overlay must have same shape (T, H, W)"
    
    gray = gray.astype(np.float32)
    red_overlay = red_overlay.astype(np.float32)

    if normalize:
        gray = (gray - np.nanmin(gray)) / (np.nanmax(gray) - np.nanmin(gray) + 1e-8)
        red_overlay = (red_overlay - np.nanmin(red_overlay)) / (np.nanmax(red_overlay) - np.nanmin(red_overlay) + 1e-8)

    # Create empty RGB output
    T, H, W = gray.shape
    rgb = np.zeros((T, H, W, 3), dtype=np.float32)
    # Red channel: base gray + red overlay
    rgb[..., 0] = np.clip(gray + red_overlay * red_scale, 0, 255)  # Red
    rgb[..., 1] = gray  # Green
    rgb[..., 2] = gray  # Blue

    return rgb

def save_cell_images(dir, cell_idx, first_frame=0, last_frame=10, frame_size=150, as_gif=False):
    """For a specified cell index, save {frame_size} x {frame_size} pixel images centred on the cell centre for the specified frames.
    Cell will be outlined in yellow, with fluorescence overlayed in red.
    """
    hdf5_file = SETTINGS.DATASET
    phase_data = np.empty((last_frame-first_frame, frame_size, frame_size))
    epi_data = np.empty((last_frame-first_frame, frame_size, frame_size))
    mask = np.empty((last_frame-first_frame, frame_size, frame_size))

    with h5py.File(hdf5_file, 'r') as f:
        x_centres, y_centres = f['Cells']['Phase']['X'][first_frame:last_frame, cell_idx], f['Cells']['Phase']['Y'][first_frame:last_frame, cell_idx]
        x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

        for idx, frame in enumerate(range(first_frame, last_frame)):
            ymin, ymax, xmin, xmax = mask_funcs.get_crop_indices((y_centres[idx], x_centres[idx]), frame_size, SETTINGS.IMAGE_SIZE)
            phase_data[idx] = np.array(f['Images']['Phase'][frame])[xmin:xmax, ymin:ymax]
            epi_data[idx] = (f['Images']['Epi'][frame][xmin:xmax, ymin:ymax])
            mask[idx] = f['Segmentations']['Phase'][frame][xmin:xmax, ymin:ymax]
            cell_mask = (mask == cell_idx)

    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')
    
    epi_data = tools.threshold_image(epi_data)
    cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()

    merged_im = overlay_red_on_gray(phase_data, epi_data, normalize=False)
    merged_im[...,0][cell_outline] = 255
    merged_im[..., 1][cell_outline] = 255

    if as_gif:
        ims = [Image.fromarray(img) for img in merged_im.astype(np.uint8)]
        ims[0].save(dir / f"cell{cell_idx}.gif", save_all=True, append_images=ims[1:], duration=200, loop=0)
    else:
        for i, im in enumerate(merged_im):
            # plt.imsave(dir / (f'{i}.jpeg'), np.transpose(im/255, (1, 2, 0)))
            plt.imsave(dir / (f'{i}.jpeg'), im/255)

# def save_cell_feature_plot(cell_idx: int, first_fame: int = 0, last_frame: int = 0) -> None:
#     plt.rcParams["font.family"] = 'serif'
#     with h5py.File(SETTINGS.DATASET, 'r') as f

def main():
    save_tracked_images(Path('temp/view'), 0, 50)
    # save_cell_images(Path('temp/view'), 1, 0, 50, as_gif=True)
    # save_cell_images(Path(r'C:\Users\php23rjb\Downloads\temp') / 'test_track', cell_idx=347, first_frame=0, last_frame=50)
    # save_masks(Path('secondwithlight_masks'), 0, SETTINGS.NUM_FRAMES)
if __name__ == '__main__':
    main()