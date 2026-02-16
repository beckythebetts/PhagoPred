from typing import Optional

import re
from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np
import imageio
import h5py
import os
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
import nd2
import tifffile
import h5py
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fill_missing_segs(seg_data: h5py.Dataset, first_appearance: np.ndarray, last_appearance: np.ndarray):
    """
    seg_data: OPen hdf5 array of shape (frames, height, width)
    num_cells: total number of cells
    first_appearance / last_appearance: lists of frames for each cell
    """
    last_known_masks = {}
    
    num_cells = len(first_appearance)

    for i in tqdm(range(seg_data.shape[0]), desc='Filling missing segmentations'):
        mask = seg_data[i, :, :]  
        filled_mask = mask.copy()

        present_ids = np.unique(mask)
        present_ids = present_ids[present_ids > 0]  # ignore background

        for cell_id in range(num_cells):
            if cell_id not in present_ids:
                if first_appearance[cell_id] <= i <= last_appearance[cell_id]:
                    if cell_id in last_known_masks:
                        reused_mask = last_known_masks[cell_id]
                        filled_mask = np.where(reused_mask, cell_id, filled_mask)
            else:
                last_known_masks[cell_id] = (mask == cell_id)

        seg_data[i, :, :] = filled_mask
        
def fill_missing_values(feature_ds: h5py.Dataset, first_appearances:np.ndarray, last_appearances: np.ndarray) -> None:
    """Interpolate the missing values for cell features"""
    for cell_idx in range(feature_ds.shape[1]):
        cell_array = feature_ds[:, cell_idx][:]
        first_appearance, last_appearance = first_appearances[cell_idx], last_appearances[cell_idx]
        valid_mask = ~np.isnan(cell_array)
        missing_idxs = np.nonzero(~valid_mask)[0]
        valid_idxs = np.nonzero(valid_mask)[0]
        missing_idxs = missing_idxs[(missing_idxs > first_appearance) & (missing_idxs < last_appearance)]
        
        try:
            cell_array[missing_idxs] = np.interp(missing_idxs, valid_idxs, cell_array[valid_idxs])
        except ValueError:
            return
    
        feature_ds[:, cell_idx] = cell_array

def fill_missing_cells(dataset: Path) -> None:
    with h5py.File(dataset, 'r+') as f:
        seg_data = f['Segmentations']['Phase']
        first_appearances = f['Cells']['Phase']['First Frame'][0]
        last_appearances = f['Cells']['Phase']['Last Frame'][0]
        
        fill_missing_segs(seg_data, first_appearances, last_appearances)
        
        for feature_name in tqdm(list(f['Cells']['Phase'].keys()), desc="Interpolating feature values"):
            if f['Cells']['Phase'][feature_name].shape[0] > 1:
                fill_missing_values(f['Cells']['Phase'][feature_name], first_appearances, last_appearances)
        
def print_cell_deaths(hdf5_file: Path = SETTINGS.DATASET) -> None:
    with h5py.File(hdf5_file, 'r') as f:
        deaths= f['Cells']['Phase']['CellDeath'][0]
        cells = np.nonzero(~np.isnan(deaths))[0]
        death_frames = deaths[cells]
        for cell, frame in zip(cells, death_frames):
            print(cell, frame)
        
        
def get_cell_end_frames(cell_idx: int, h5py_file: h5py.File) -> tuple[int, int]:
    """Use 'X' coord datset to get frame of first and last non nan"""
    not_nan_mask = ~np.isnan(h5py_file['Cells']['Phase']['X'][:, cell_idx])
    first_frame = np.argmax(not_nan_mask)
    last_frame = len(not_nan_mask) - 1 - np.argmax(not_nan_mask[::-1])
    return first_frame, last_frame

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

def hdf5_from_nd2(nd2_file: Path, hdf5_file: Path, phase_channel: int=1, epi_channel: int=0, start_frame: int=0, end_frame: Optional[int] = None) -> None:
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)
    with nd2.ND2File(nd2_file) as n:
        with h5py.File(hdf5_file, 'w') as h:
            Images = h.create_group('Images')
        # ****** Image MetaData ********
            Images.attrs['Camera'] = 'Andor neo 5.5'
            Images.attrs['Pixel Size / um'] = 6.5
            Images.attrs['Objective magnification'] = 20
            Images.attrs['Image size / pixels'] = [1430, 1750]
            Images.attrs['Phase exposure / ms'] = 33
            Images.attrs['Epi exposure / ms'] = 200
            Images.attrs['Time interval / s'] = 60
            Images.attrs['Number of frames'] = end_frame-start_frame
            Images.attrs['Filter'] = False
            Images.attrs['NA'] = 0.45
            Images.attrs['Notes'] = 'Macrophages and staph bacteria, but no fluorescence'

            Images.attrs['Resolution / um'] = Images.attrs['Pixel Size / um'] / Images.attrs['Objective magnification']
            Images.attrs['FOV / um'] = Images.attrs['Image size / pixels'] * Images.attrs['Resolution / um']

            Phase = Images.create_group('Phase')
            Epi = Images.create_group('Epi')

            for frame_idx in range(start_frame, end_frame):
                sys.stdout.write(f'\rFrame {frame_idx+1} / {end_frame}')
                sys.stdout.flush()
                frame_arr = np.array(n.read_frame(frame_idx))
                phase_im = frame_arr[phase_channel]
                epi_im = frame_arr[epi_channel]
                for im, group in zip((phase_im, epi_im), (Phase, Epi)):
                    im = (min_max_normalise(im) * 256).astype(np.uint8)
                    group.create_dataset(f'{frame_idx:04}', data=im)


def natural_sort_key(s):
    parts = re.split(r'(\d+)', str(s))
    return [int(part) if part.isdigit() else part.lower() for part in parts]

def hdf5_from_tiffs(tiff_files_path: Path, hdf5_file: Path,
                    phase_channel: int = 1, epi_channel: int = 0,
                    batch_size: int = 100) -> None:
    """Convert multi-file, multi-page .ome.tif files to HDF5 with batched reading and processing."""

    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)

    tiff_files = sorted(tiff_files_path.glob("*.ome.tif"), key=natural_sort_key)
    print(tiff_files)
    if not tiff_files:
        raise ValueError("No .ome.tif files found.")

    # Get shape info from first file
    with tifffile.TiffFile(str(tiff_files[0])) as tif:
        series = tif.series[0]
        shape = series.shape
        T, C, Y, X = shape

    # Create HDF5 file and datasets (extendable)
    with h5py.File(hdf5_file, 'w') as h:
        Images = h.create_group('Images')
        Images.attrs['Camera'] = "Orca Flash 4.0, C11440"
        Images.attrs['Pixel Size / um'] = 6.5
        Images.attrs['Image size / pixels'] = [Y, X]
        Images.attrs['Objective magnification'] = 20
        res_um = Images.attrs['Pixel Size / um'] / Images.attrs['Objective magnification']
        Images.attrs['Resolution / um'] = res_um
        Images.attrs['FOV / um'] = [Y * res_um, X * res_um]
        Images.attrs['Phase exposure / ms'] = 1000
        Images.attrs['Epi exposure / ms'] = 10000
        Images.attrs['Time interval / s'] = 60
        Images.attrs['Objective NA'] = 0.45
        Images.attrs['Number of frames'] = 0  # to be updated
        Images.attrs['Notes'] = "J774 macrphages, SH1000 S.Aureus MCherry, grown in TSB + 0.1% Tet overnight. Imaged in DMEM. 1*10**5 cells per dish 1:1 MOI."

        Phase = Images.create_group('Phase')
        Epi = Images.create_group('Epi')

        phase_ds = Phase.create_dataset("Data", shape=(0, Y, X), maxshape=(None, Y, X),
                                        dtype='uint8', chunks=(1, Y, X))
        epi_ds = Epi.create_dataset("Data", shape=(0, Y, X), maxshape=(None, Y, X),
                                    dtype='uint8', chunks=(1, Y, X))

        total_frames = 0
        pages_per_file = []

        # First, gather number of pages per file to manage global indexing
        for file in tiff_files:
            with tifffile.TiffFile(str(file)) as tif:
                pages_per_file.append(len(tif.pages))

        # Iterate files and read batches of pages correctly by local indexing
        for i, file in enumerate(tiff_files):
            sys.stdout.write(f"\rProcessing file {i + 1} / {len(tiff_files)}; total frames: {total_frames}")
            sys.stdout.flush()
            with tifffile.TiffFile(str(file)) as tif:
                pages = tif.pages
                num_pages = len(pages)
                num_channels = C  # or get from series shape as needed

                for start_idx in range(0, num_pages // num_channels, batch_size):
                    end_idx = min(start_idx + batch_size, num_pages // num_channels)
                    current_batch_size = end_idx - start_idx

                    phase_batch = np.empty((current_batch_size, Y, X), dtype=np.uint16)
                    epi_batch = np.empty((current_batch_size, Y, X), dtype=np.uint16)

                    for t_idx in range(current_batch_size):
                        frame_idx = start_idx + t_idx
                        phase_page_idx = frame_idx * num_channels + phase_channel
                        epi_page_idx = frame_idx * num_channels + epi_channel

                        phase_batch[t_idx] = pages[phase_page_idx].asarray()
                        epi_batch[t_idx] = pages[epi_page_idx].asarray()

                    # Convert batches to uint8 with per-frame min-max scaling
                    phase_uint8 = batch_convert_to_uint8(phase_batch)
                    epi_uint8 = batch_convert_to_uint8(epi_batch)

                    # Resize datasets
                    phase_ds.resize((total_frames + current_batch_size, Y, X))
                    epi_ds.resize((total_frames + current_batch_size, Y, X))

                    # Write batches
                    phase_ds[total_frames:total_frames + current_batch_size] = phase_uint8
                    epi_ds[total_frames:total_frames + current_batch_size] = epi_uint8

                    total_frames += current_batch_size

            

        Images.attrs['Number of frames'] = total_frames
        print(f"\nHDF5 creation complete. {total_frames} frames written.")


def convert_to_uint8(image: np.ndarray, min_val=None, max_val=None) -> np.ndarray:
    """
    Convert a 16-bit image (or higher bit depth) to 8-bit using linear contrast stretching.
    If min_val and max_val are not given, they're computed from the image.
    """
    if min_val is None:
        min_val = image.min()
    if max_val is None:
        max_val = image.max()

    if min_val == max_val:
        # Prevent division by zero if the image is flat
        return np.zeros_like(image, dtype=np.uint8)

    scaled = (image.astype(np.float32) - min_val) / (max_val - min_val)
    return np.clip(scaled * 255, 0, 255).astype(np.uint8)

def batch_convert_to_uint8(images: np.ndarray, min_val=None, max_val=None) -> np.ndarray:
    """
    Convert a batch of images (T, Y, X) to 8-bit using linear contrast stretching.
    If min_val/max_val are None, compute per-frame min and max.

    Returns array of shape (T, Y, X), dtype=uint8.
    """
    images = images.astype(np.float32)

    if min_val is None:
        min_val = images.reshape(images.shape[0], -1).min(axis=1).reshape(-1, 1, 1)
    else:
        min_val = np.array(min_val, dtype=np.float32).reshape(-1, 1, 1)

    if max_val is None:
        max_val = images.reshape(images.shape[0], -1).max(axis=1).reshape(-1, 1, 1)
    else:
        max_val = np.array(max_val, dtype=np.float32).reshape(-1, 1, 1)

    # Avoid division by zero: where max == min, set range to 1
    scale = np.where(max_val != min_val, max_val - min_val, 1)

    scaled = (images - min_val) / scale
    scaled = np.clip(scaled * 255, 0, 255)

    # Set flat images (min == max) to 0
    flat_mask = (max_val == min_val).reshape(-1)
    if np.any(flat_mask):
        scaled[flat_mask] = 0

    return scaled.astype(np.uint8)

# def batch_convert_to_uint8(
#     images: np.ndarray,
#     lower_percentile: float = 1.0,
#     upper_percentile: float = 99.5
# ) -> np.ndarray:
#     """
#     Convert a batch of images (T, Y, X) to 8-bit using contrast stretching.
#     Uses percentile-based min/max to avoid outliers like hot pixels.

#     Parameters:
#         images: (T, Y, X) array
#         lower_percentile: percentile used as minimum (e.g., 1%)
#         upper_percentile: percentile used as maximum (e.g., 99.5%)

#     Returns:
#         uint8 image array of shape (T, Y, X)
#     """
#     images = images.astype(np.float32)

#     # Compute per-frame percentiles
#     T = images.shape[0]
#     flat_images = images.reshape(T, -1)

#     min_vals = np.percentile(flat_images, lower_percentile, axis=1).reshape(-1, 1, 1)
#     max_vals = np.percentile(flat_images, upper_percentile, axis=1).reshape(-1, 1, 1)

#     # Avoid division by zero
#     scale = np.where(max_vals != min_vals, max_vals - min_vals, 1)
#     scaled = (images - min_vals) / scale
#     scaled = np.clip(scaled * 255, 0, 255)

#     # Set flat images to zero
#     flat_mask = (max_vals == min_vals).reshape(-1)
#     if np.any(flat_mask):
#         scaled[flat_mask] = 0

#     return scaled.astype(np.uint8)


def min_max_normalise(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def standardise(image):
    if isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        mean = np.mean(image)
        std = np.std(image)
    
    elif isinstance(image, torch.Tensor):
        image = image.to(torch.float32)
        mean = torch.mean(image)
        std = torch.std(image)
    
    else:
        raise TypeError('Expects np.ndarray or torch.Tensor')

    standardized_image = (image - mean) / std

    return standardized_image

# def show_segmentation(image_array, mask_array, true_mask_array=None):
#     # Ensure RGB format
#     if len(image_array.shape) == 2:
#         image_array = np.stack((image_array,) * 3, axis=-1)
    
#     # Convert to torch tensor
#     image_array = torch.tensor(image_array).clone()
#     mask_array = torch.tensor(mask_array)
    
#     # Create predicted outline
#     pred_outline = mask_funcs.mask_outline((mask_array > 0).int(), thickness=2)
    
#     if true_mask_array is not None:
#         true_mask_array = torch.tensor(true_mask_array)
#         true_outline = mask_funcs.mask_outline((true_mask_array > 0).int(), thickness=2)

#         # Compute overlap
#         overlap = pred_outline & true_outline
#         only_pred = pred_outline & ~overlap
#         only_true = true_outline & ~overlap

#         # Make sure image is in uint8 range
#         if image_array.dtype != torch.uint8:
#             image_array = image_array.clamp(0, 255).byte()

#         # Apply colors:
#         # Yellow for overlap (255, 255, 0)
#         image_array[overlap] = torch.tensor([255, 255, 0], dtype=torch.uint8)

#         # Red for predicted only (255, 0, 0)
#         image_array[only_pred] = torch.tensor([255, 0, 0], dtype=torch.uint8)

#         # Green for true only (0, 255, 0)
#         image_array[only_true] = torch.tensor([0, 255, 0], dtype=torch.uint8)

#     else:
#         # No true mask, only show predicted outline in red
#         image_array[pred_outline] = torch.tensor([255, 0, 0], dtype=torch.uint8)

#     return image_array.cpu().numpy()    

def show_segmentation(image_array, mask_array, true_mask_array=None, thickness=3):
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)

    image_tensor = torch.tensor(image_array).clone()
    if image_tensor.dtype != torch.uint8:
        image_tensor = image_tensor.clamp(0, 255).byte()

    # Handle multiple predicted masks: [N, H, W] or [H, W]
    mask_tensor = torch.tensor(mask_array)
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.sum(dim=0).int()
    else:
        mask_tensor = mask_tensor.int()

    pred_outline = mask_funcs.mask_outlines(mask_tensor, thickness=thickness)

    if true_mask_array is not None:
        true_tensor = torch.tensor(true_mask_array)
        if true_tensor.ndim == 3:
            true_tensor = true_tensor.sum(dim=0).int()
        else:
            true_tensor = true_tensor.int()

        true_outline = mask_funcs.mask_outlines(true_tensor, thickness=thickness)

        # Create masks for outline positions
        overlap = (pred_outline > 0) & (true_outline > 0)
        only_pred = (pred_outline > 0) & (~overlap)
        only_true = (true_outline > 0) & (~overlap)

        # Draw order: green (true), red (pred), yellow (overlap)
        image_tensor[only_true] = torch.tensor([0, 255, 0], dtype=torch.uint8)       # Green
        image_tensor[only_pred] = torch.tensor([255, 0, 0], dtype=torch.uint8)       # Red
        image_tensor[overlap]    = torch.tensor([255, 255, 0], dtype=torch.uint8)    # Yellow

    else:
        # Only predicted outline
        image_tensor[pred_outline > 0] = torch.tensor([255, 0, 0], dtype=torch.uint8)

    return image_tensor.cpu().numpy()
# def show_segmentation(image_array, mask_array, true_mask_array=None):
#     if len(image_array.shape) == 2:
#         image_array = np.stack((image_array,) * 3, axis=-1)

#     image_tensor = torch.tensor(image_array).clone()
#     if image_tensor.dtype != torch.uint8:
#         image_tensor = image_tensor.clamp(0, 255).byte()

#     mask_tensor = torch.tensor(mask_array).int()

#     # Get predicted outline
#     pred_outline = mask_funcs.mask_outline((mask_tensor > 0), thickness=2)

#     if true_mask_array is not None:
#         true_tensor = torch.tensor(true_mask_array).int()
#         true_outline = mask_funcs.mask_outline((true_tensor > 0), thickness=2)

#         # Keep all outline pixels (don't remove overlaps yet)
#         full_pred_outline = pred_outline.clone()
#         full_true_outline = true_outline.clone()

#         # Compute overlap AFTER outlines have full thickness
#         overlap = full_pred_outline & full_true_outline

#         # Areas unique to each
#         only_pred = full_pred_outline & ~overlap
#         only_true = full_true_outline & ~overlap

#         # Draw order: true (green), pred (red), then overlap (yellow) to overwrite shared pixels

#         # Green for true outline
#         image_tensor[only_true] = torch.tensor([0, 255, 0], dtype=torch.uint8)

#         # Red for predicted outline
#         image_tensor[only_pred] = torch.tensor([255, 0, 0], dtype=torch.uint8)

#         # Yellow for overlap
#         image_tensor[overlap] = torch.tensor([255, 255, 0], dtype=torch.uint8)

#     else:
#         # Only predicted outline
#         image_tensor[pred_outline] = torch.tensor([255, 0, 0], dtype=torch.uint8)

#     return image_tensor.cpu().numpy()
                                  
# def show_segmentation(image_array, mask_array, true_mask_array=None):
#     if len(image_array.shape) == 2:
#         image_array = torch.tensor(np.stack((image_array, image_array, image_array), axis=-1))
#     else:
#         image_array = torch.tensor(image_array)
#     mask_array = torch.tensor(mask_array)
#     outline = mask_funcs.mask_outline(torch.where(mask_array>0, 1, 0), thickness=2)

#     image_array[:,:,0][outline] = torch.max(image_array)
#     if true_mask_array is not None:
#         true_mask_array = torch.tensor(true_mask_array)
#         true_outline = mask_funcs.mask_outline(torch.where(true_mask_array>0, 1, 0), thickness=2)
#         image_array[:, :, 1][true_outline] = torch.max(image_array)
#     return image_array.cpu().numpy()

def show_semantic_segmentation(image_array, mask_array, true_mask_array=None):
    if len(image_array.shape) == 2:
        image_array = torch.stack((image_array, image_array, image_array), dim=-1)
    # elif len(image_array.shape) == 4:
    #     image_array = image_array[0, 0, :, :]
    #     image_array = torch.stack((image_array, image_array, image_array), dim=-1)

    if true_mask_array is None:
        image_array[:,:,1][mask_array] = 0
        image_array[:,:,2][mask_array] = 0
    else:
        image_array[:,:,2][mask_array | true_mask_array] = 0
        # image_array[:,:,2] = image_array[:,:,2].masked_fill(mask_array & true_mask_array, 0)
        image_array[:,:,1][mask_array & ~true_mask_array] = 0
        image_array[:,:,0][true_mask_array & ~mask_array] = 0

    return image_array*255

def repack_hdf5(file=SETTINGS.DATASET):
    if isinstance(file, str):
        file = Path(file)
    with h5py.File(file, 'r') as f:
        all_groups = list(f.keys())
    temp_file = file.parent / (file.stem + '_temp' + file.suffix) 
    copy_hdf5_groups(file, temp_file, all_groups)
    os.remove(file)
    temp_file.rename(file)

def copy_hdf5_groups(src_filename, dest_filename, groups_to_copy):
    """
    Copy specified groups and all their attributes and datasets from a source HDF5 file to a new HDF5 file.
    
    Parameters:
    - src_filename: str, the path to the source HDF5 file.
    - dest_filename: str, the path to the destination HDF5 file.
    - groups_to_copy: list of str, list of group paths to copy from the source to the destination file.
    """
    with h5py.File(src_filename, 'r') as src_file, h5py.File(dest_filename, 'w') as dest_file:
        def copy_group(src_group, dest_group):
            for attr_name, attr_value in src_group.attrs.items():
                dest_group.attrs[attr_name] = attr_value
            
            for name, item in src_group.items():
                if isinstance(item, h5py.Dataset):  # Copy datasets directly
                    maxshape = item.maxshape
                    dset = dest_group.create_dataset(                        
                        name,
                        shape=item.shape,
                        maxshape=item.maxshape,
                        dtype=item.dtype,
                        chunks=item.chunks)
                
                    if item.chunks:
                        for i in tqdm(range(0, item.shape[0], item.chunks[0]), desc=f'Copying {name} dataset'):
                            sel = slice(i, min(i + item.chunks[0], item.shape[0]))
                            dset[sel, ...] = item[sel, ...]
                    else:
                        dset[...] = item[()]
                    for attr_name, attr_value in item.attrs.items():
                        dest_group[name].attrs[attr_name] = attr_value
                elif isinstance(item, h5py.Group):  # Recurse into subgroups
                    subgroup = dest_group.create_group(name)
                    copy_group(item, subgroup)
        
        for group_name in groups_to_copy:
            if group_name in src_file:
                src_group = src_file[group_name]
                dest_group = dest_file.create_group(group_name)
                copy_group(src_group, dest_group)
            else:
                print(f"Warning: Group '{group_name}' not found in the source file.")

def crop_hdf5(file, crop_size=[2048, 2048]):
    """
    Image datasets for each frame must have been created with maxshape=(None, None)
    """
    with h5py.File(file, 'r+') as f:
        x_centre, y_centre = (int(f['Images'].attrs['Image size / pixels'][0] / 2),
                              int(f['Images'].attrs['Image size / pixels'][1] / 2))
                        
        f['Images'].attrs['Image size / pixels'] = crop_size
        
        for modality in ('Phase', 'Epi'):
            for frame in list(f['Images'][modality].keys()):
                cropped_im  = f['Images'][modality][frame][int(x_centre - crop_size[0]/2):int(x_centre + crop_size[0]/2), 
                                                           int(y_centre - crop_size[1]/2):int(y_centre + crop_size[1]/2)]
                f['Images'][modality][frame].resize(crop_size)
                f['Images'][modality][frame][()] = cropped_im


def create_hdf5(hdf5_filename, phase_tiffs_path, epi_tiffs_path):
    if os.path.exists(hdf5_filename):
        os.remove(hdf5_filename)
    with h5py.File(hdf5_filename, 'w') as f:
        Images = f.create_group('Images')
        # ****** Image MetaData ********
        Images.attrs['Camera'] = 'NA - 15percent - Ash'
        Images.attrs['Pixel Size / um'] = 1
        Images.attrs['Objective magnification'] = 1
        Images.attrs['Image size / pixels'] = [1200,1200]
        Images.attrs['Phase exposure / ms'] = 1
        Images.attrs['Epi exposure / ms'] = 1
        Images.attrs['Time interval / s'] = 10
        Images.attrs['Number of frames'] = 300
        Images.attrs['Filter'] = False
        Images.attrs['NA'] = 1

        Images.attrs['Resolution / um'] = Images.attrs['Pixel Size / um'] / Images.attrs['Objective magnification']
        Images.attrs['FOV / um'] = Images.attrs['Image size / pixels'] * Images.attrs['Resolution / um']

        Phase = Images.create_group('Phase')
        Epi = Images.create_group('Epi')
        print('\nConverting Phase Images\n')
        i=1
        for im in Path(phase_tiffs_path).iterdir():
            sys.stdout.write(f'\rFrame {i} / {Images.attrs["Number of frames"]}')
            sys.stdout.flush()
            Phase.create_dataset(im.stem, data=np.array(Image.open(im, mode='r')))
            i += 1
        print('\nConverting Epi Images\n')
        i = 1
        for im in Path(epi_tiffs_path).iterdir():
            sys.stdout.write(
                f'\rFrame {i} / {Images.attrs["Number of frames"]}')
            sys.stdout.flush()
            Epi.create_dataset(im.stem, data=np.array(Image.open(im, mode='r')))
            i += 1



def rename_datasets_in_group(h5_file_path, group_path):
    """
    Renames all datasets within a specified group in an HDF5 file, removing the first 13 characters from their names.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
        group_path (str): Path to the group containing the datasets to rename.

    Notes:
        This operation creates new datasets with the updated names and deletes the old ones.
    """
    with h5py.File(h5_file_path, 'r+') as h5_file:
        group = h5_file[group_path]

        # Collect original dataset names
        original_dataset_names = [name for name in group if isinstance(group[name], h5py.Dataset)]

        for old_name in original_dataset_names:
            # Compute the new name by removing the first 13 characters
            new_name = old_name[-4:]

            if new_name == "":
                raise ValueError(f"The dataset name '{old_name}' would become empty after renaming.")

            # Copy the old dataset to a new dataset with the updated name
            group.copy(old_name, new_name)

            # Delete the old dataset
            del group[old_name]



def get_images(h5file, mode, first_frame=0, last_frame=SETTINGS.NUM_FRAMES):
     return np.array([h5file['Images'][mode][frame][:] for frame in list(h5file['Images'][mode].keys())[first_frame:last_frame]], dtype='uint8')

def get_masks(h5file, mode, first_frame=0, last_frame=SETTINGS.NUM_FRAMES):
     return np.array([h5file['Segmentations'][mode][frame][:] for frame in list(h5file['Segmentations'][mode].keys())[first_frame:last_frame]], dtype='int16')

def remake_dir(path):
    if path.is_dir():
        check = input(f'Delete directory {str(path)}? [y,n] ')
        if check.lower() != 'y':
            raise SystemExit(0)
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path

def unique_nonzero(array, return_counts=False):
    return np.unique(array[array != 0], return_counts=return_counts)

def min_max_scale(image):
    min = np.min(image)
    max = np.max(image)
    return (image - min) / (max - min)


def torch_min_max_scale(image):
    min = torch.min(image)
    max = torch.max(image)
    return (image - min) / (max - min)


def read_tiff(path):
    im = Image.open(path, mode='r')
    return np.array(im)


def save_tiff(array, path):
    imageio.imwrite(str(path), array)
    # im = Image.fromarray(array)
    # im.save(path)


def draw_line(array, x0, x1, y0, y1, colour):
    x0, x1, y0, y1 = x0.round(), x1.round(), y0.round(), y1.round()
    transpose = abs(x1 - x0) < abs(y1 - y0)
    if transpose:
        array = torch.permute(array, (1, 0, 2))
        x0, y0, x1, y1 = y0, x0, y1, x1
    if (x0-x1).round()==0 and (y0-y1).round()==0:
        return array
    if x0 > x1:
        x0, x1, y0, y1 = x1, x0, y1, y0

    x = torch.arange(x0, x1).to(device)
    y = ((y1-y0)/(x1-x0))*(x-x0) + y0

    x = torch.clamp(x.round().to(int), min=0, max=array.shape[0] - 1)
    y = torch.clamp(y.round().to(int), min=0, max=array.shape[1] - 1)

    array[x, y, :] = colour
    # print(array.shape)
    return array if not transpose else torch.permute(array, (1, 0, 2))


def split_list_into_sequences(the_list, return_indices=False):
    if return_indices == False:
        sequences = [[the_list[0]]]
        for i, list_item in enumerate(the_list[1:]):
            if list_item - the_list[i] <= SETTINGS.FRAME_MEMORY:
                sequences[-1].append(list_item)
            else:
                sequences.append([list_item])
        return sequences
    else:
        sequences = [[0]]
        for i, list_item in enumerate(the_list[1:]):
            if list_item - the_list[i] <= SETTINGS.FRAME_MEMORY:
                sequences[-1].append(i+1)
            else:
                sequences.append([i+1])
        return sequences
    
def append_hdf5(f, array):
    f.resize((len(f)+len(array),))
    f[-len(array):-1] = array

def get_features_ds(f, feature):
    column_idx = list(f.attrs['features']).index(feature)
    return f[:, :, column_idx]
    
def fill_nans(array):
    s = pd.Series(array)
    # filled_s = s.fillna(method='ffill').fillna(method='bfill')
    filled_s = s.ffill().bfill()
    return filled_s.to_numpy()

def threshold_image(frames: np.ndarray, lower_percentile: float = 99, upper_percentile: float = 99.9) -> np.ndarray:
    """Calculate percentiles for given frames and threshold image.

    Arguments
    ---------
    frames: [N, X, Y] np.ndarray
    """
    lower_threshold = np.percentile(frames.flatten(), lower_percentile)
    upper_threshold = np.percentile(frames.flatten(), upper_percentile)

    return np.where((frames>=lower_threshold) & (frames<=upper_threshold), frames, 0)


def plot_tracks(save_as):
    tracks_plot = torch.zeros((*SETTINGS.IMAGE_SIZE, 3), dtype=torch.uint8).to(device)
    tracks_plot = torch.permute(tracks_plot, (1, 0, 2))
    print('\nPLOTTING TRACKS\n')
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        xs = get_features_ds(f['Cells']['Phase'], 'x')
        ys = get_features_ds(f['Cells']['Phase'], 'y')
        for cell_idx in range(1, xs.shape[1]):
            colour = torch.tensor(np.random.uniform(0, 2 ** (8) - 1, size=3), dtype=torch.uint8).to(device)
            # xcentres = torch.tensor(f['Features'][cell]['MorphologicalFeatures']['xcentre']).to(device)
            # ycentres = torch.tensor(f['Features'][cell]['MorphologicalFeatures']['ycentre']).to(device)
            xsi = torch.tensor(xs[:, cell_idx]).to(device)
            ysi = torch.tensor(ys[:, cell_idx]).to(device)
            xsi, ysi = xsi[~torch.isnan(xsi)], ysi[~torch.isnan(ysi)]
            for i in range(len(xsi) - 1):
                tracks_plot = draw_line(tracks_plot, xsi[i], xsi[i+1], ysi[i], ysi[i+1], colour)
    print(tracks_plot.shape)
    imageio.imwrite(save_as, tracks_plot.cpu())

def crop_bbox_to_fit(bbox, crop_x, crop_y, crop_width, crop_height):
    """
    Crop the bounding box to fit within the crop boundaries.
    bbox: [x, y, width, height]
    crop_x, crop_y: Top-left corner of the cropped area
    crop_width, crop_height: Size of the cropped area
    """
    cropped_x = max(bbox[0], crop_x) - crop_x
    cropped_y = max(bbox[1], crop_y) - crop_y
    cropped_w = min(bbox[0] + bbox[2], crop_x + crop_width) - max(bbox[0], crop_x)
    cropped_h = min(bbox[1] + bbox[3], crop_y + crop_height) - max(bbox[1], crop_y)
    
    # If the cropped width or height is zero or negative, the bbox doesn't intersect with the crop
    if cropped_w <= 0 or cropped_h <= 0:
        return None
    
    return [cropped_x, cropped_y, cropped_w, cropped_h]

def split_coco_by_category(coco_file, save_dir, category=None):
    """
    Create separate coco_json file for each category.
    """
    with open(coco_file, 'r') as f:
        coco = json.load(f)
    categories = coco['categories'] if category is None else [cat for cat in coco['categories'] if cat['name'] == category]
    for category in categories:
        new_coco_file_name = save_dir / f'{coco_file.stem}_{category["name"]}_{coco_file.suffix}' if category is None else save_dir / coco_file.name
        new_coco = {
            'images': coco['images'],
            'annotations': [],
            'categories': [{'id': 1, 'name': category['name']}]
        }
        for annotation in coco['annotations']:
            if annotation['category_id'] == category['id']:
                new_annotation = annotation.copy()
                new_annotation['category_id'] = 1
                new_coco['annotations'].append(new_annotation)
        with open(new_coco_file_name, 'w') as f:
            json.dump(new_coco, f)

def crop_polygon_to_fit(polygon, crop_x, crop_y, crop_width, crop_height):
    """
    Crop the polygon coordinates to fit within the crop boundaries.
    polygon: List of [x, y] pairs
    crop_x, crop_y: Top-left corner of the cropped area
    crop_width, crop_height: Size of the cropped area
    """
    cropped_polygon = []
    for i in range(0, len(polygon), 2):
        x, y = polygon[i], polygon[i + 1]
        
        # Check if the point is within the crop
        if crop_x <= x < crop_x + crop_width and crop_y <= y < crop_y + crop_height:
            cropped_polygon.append(x - crop_x)
            cropped_polygon.append(y - crop_y)
    
    return cropped_polygon if cropped_polygon else None

def split_image_and_coco(image_dir, coco_path, new_image_dir, new_coco_path, x_splits=2, y_splits=2):
    """
    Splits images and coco annotations into smaller sections.
    """
    remake_dir(new_image_dir)
    
    # Load the original COCO annotations
    with open(coco_path, mode='r') as f:
        coco = json.load(f)
    
    # Prepare for new coco annotations
    new_coco = {
        'images': [],
        'annotations': [],
        'categories': coco['categories']
    }
    
    # Keep track of the annotation id
    new_annotation_id = 0
    
    for image in coco['images']:
        file_name = Path(image['file_name'])
        im_array = plt.imread(image_dir / file_name)
        
        # Image dimensions and split sizes
        x_size = int(image['width'] / x_splits)
        y_size = int(image['height'] / y_splits)
        
        # Split the image into x_splits by y_splits sections
        x_split_coords = [i * x_size for i in range(x_splits)]
        y_split_coords = [i * y_size for i in range(y_splits)]
        
        # Process each split section
        for i, x in enumerate(x_split_coords):
            for j, y in enumerate(y_split_coords):
                # Crop the image
                cropped_image = im_array[y:y + y_size, x:x + x_size]
                
                # Save the cropped image
                cropped_image_name = f'{file_name.stem}_{i}{j}{file_name.suffix}'
                plt.imsave(new_image_dir / cropped_image_name, cropped_image, cmap='gray')
                
                # Add the new image entry to the coco
                new_image = {
                    'id': len(new_coco['images']) + 1,
                    'file_name': cropped_image_name,
                    'width': x_size,
                    'height': y_size
                }
                new_coco['images'].append(new_image)
                
                # Update the annotations (bounding boxes and polygons)
                for annotation in coco['annotations']:
                    if annotation['image_id'] == image['id']:
                        # Check if the annotation is within or partially within the cropped section
                        bbox = annotation['bbox']
                        # Crop the bounding box to fit within the cropped image
                        cropped_bbox = crop_bbox_to_fit(bbox, x, y, x_size, y_size)
                        
                        if cropped_bbox:
                            # Update the annotation with the cropped bbox
                            new_annotation = annotation.copy()
                            new_annotation['id'] = new_annotation_id
                            new_annotation['image_id'] = len(new_coco['images'])  # Associate with the new image
                            new_annotation['bbox'] = cropped_bbox
                            new_annotation['bbox_mode'] = 0
                            
                            # Update the segmentation polygons
                            if 'segmentation' in new_annotation:
                                new_polygon = []
                                for poly in new_annotation['segmentation']:
                                    cropped_poly = crop_polygon_to_fit(poly, x, y, x_size, y_size)
                                    # if len(cropped_poly) == 4:
                                    #     cropped_poly.extend(cropped_poly[-2:])
                                    if cropped_poly:
                                        new_polygon.append(cropped_poly)
                                new_annotation['segmentation'] = new_polygon
                            
                            # If the annotation has a cropped bbox or polygon, add it to the new COCO
                            if new_annotation.get('bbox') and new_annotation.get('segmentation'):
                                if len(new_annotation['segmentation'][0]) > 4:
                                    new_coco['annotations'].append(new_annotation)
                                    new_annotation_id += 1
    
    # Save the new coco annotations to file
    with open(new_coco_path, 'w') as f:
        json.dump(new_coco, f)

import os

def rename_files_in_folder(folder_path):
    # Loop through all files in the given folder
    for filename in os.listdir(folder_path):
        # Get the full path to the file
        full_path = os.path.join(folder_path, filename)
        
        # Check if it's a file and not a directory
        if os.path.isfile(full_path):
            # Split the filename into the name and the extension
            name, ext = os.path.splitext(filename)
            
            # Find the position where digits start and where they end in the name
            # Split into the prefix (letters) and suffix (digits)
            for i in range(len(name)):
                if name[i].isdigit():
                    prefix = name[:i]
                    suffix = name[i:]
                    break
            
            # Create the new name by switching prefix and suffix
            new_name = suffix + prefix + ext
            
            # Get the full path of the new file
            new_full_path = os.path.join(folder_path, new_name)
            
            # Rename the file
            os.rename(full_path, new_full_path)
            print(f'Renamed: {filename} -> {new_name}')

def add_empty_epi(file=SETTINGS.DATASET):
    with h5py.File(file, 'r+') as f:
        # for group in ('Images', 'Segmentations'):
        #     for frame in list(f[group]['Phase'].keys()):
        #         f[group].create_dataset(f'Epi/{frame}', shape=f[group]['Phase'][frame].shape, maxshape=(None, None))
        # f['Cells'].create_dataset('Epi', shape=(0,0,3), maxshape=(None, None, None))
        # f['Cells']['Epi'].resize((1,1,3))
        f['Cells']['Epi'].attrs['features'] = ['x', 'y', 'area']


def coco_to_semantic_training_data(image_dir, coco_json, target_im_dir, target_mask_dir):
    remake_dir(target_im_dir)
    remake_dir(target_mask_dir)
    for im in image_dir.iterdir():
        shutil.copy(im, target_im_dir / im.name)
        mask = mask_funcs.coco_to_binary_mask(coco_json, im)
        plt.imsave(target_mask_dir / im.name, mask, cmap='gray')

def calculate_padding(dim, multiple=32):
    """
    Calculate padding needed to make the dimension divisible by the target multiple.
    
    Args:
        dim (int): The current dimension (height or width) of the image.
        multiple (int): The target multiple to which the dimension should be divisible (default: 32).
        
    Returns:
        int: The padding required to make the dimension divisible by the multiple.
    """
    remainder = dim % multiple
    if remainder == 0:
        return 0  # No padding needed if it's already divisible
    return multiple - remainder  # Padding required to reach the next multiple

def pad_image(image, target_multiple=32):
    '''
    image: pytorch tensor, dims[C, H, W]
    '''
    # print(image.shape)
    height, width = image.shape[1], image.shape[2]
    padding_height = calculate_padding(height, target_multiple)
    padding_width = calculate_padding(width, target_multiple)

    padding = (0, padding_width, 0, padding_height)

    padded_image = torch.nn.functional.pad(image, padding)

    return padded_image, padding_width, padding_height

def unpad_image(image, padding_width, padding_height):
    unpad = lambda padding: slice(None) if padding == 0 else slice(0, -padding)
    return image[unpad(padding_height), unpad(padding_width)]

def remove_outliers(arr):
    '''
    using interquartile range (+/- 1.5*q)
    '''
    no_nan = arr[~np.isnan(arr)]
    q1 = np.percentile(no_nan, 0.25)
    q3 = np.percentile(no_nan, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    arr[(arr < lower_bound)] = np.nan
    arr[(arr > upper_bound)] = np.nan
    return arr

def corr_coefficient(array1, array2):
    '''
    deals with np.nan when calculating correlation coefficeinet (numpy)
    '''
    mask = ~np.isnan(array1) & ~np.isnan(array2)
    return np.corrcoef(array1[mask], array2[mask])[0, 1]

def get_hist(feature):
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        values = get_features_ds(f['Cells']['Phase'], feature).flatten()
        # values[values > 0.01] = 0
        plt.rcParams["font.family"] = 'serif'
        histogram = plt.hist(values, bins=100)
        plt.xlabel(feature)
        plt.show()
    return histogram, feature

def plot_correlations(features='all'):

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        features_df = f['Cells']['Phase'][:]
        features_df = np.abs(features_df)
        all_features = f['Cells']['Phase'].attrs['features']
        if features == 'all':
            feature_idxs = np.arange(len(all_features))
            features = all_features
        else:
            feature_idxs = np.where(np.isin(all_features, features))[0]

        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(len(feature_idxs), len(feature_idxs))
        cmap = plt.get_cmap('viridis_r')
        # cmap = plt.get_cmap('prism')
        
        for i, f_i in enumerate(feature_idxs):
            for j, f_j in enumerate(feature_idxs):

                features_i, features_j = remove_outliers(features_df[:, :, f_i].flatten()), remove_outliers(features_df[:, :, f_j].flatten())
                R = corr_coefficient(features_i, features_j)
                color = cmap((R+1)/2)

                #set plot titles and axis labels

                if i == 0:
                    axs[i, j].set_title(features[j])

                if j == 0:
                    axs[i, j].set_ylabel(features[i], rotation = 45, labelpad=30)
                
                if j!=0 and i!=j:
                    axs[i, j].set_yticklabels([])
                
                if i != len(features)-1:
                    axs[i, j].set_xticklabels([])

                # plot

                if i > j:
                    axs[i, j].scatter(features_j, features_i, s=0.1, linewidths=0, color=color)
                    # axs[i, j].grid()

                if i == j:
                    axs[i, j].hist(features_i, bins=100, color='k')
                    # axs[i, j].grid()

                if i < j:
                    axs[i, j].text(0.5, 0.5, f'R = {R:.2f}', ha='center', va='center', transform=axs[i, j].transAxes,
                                    fontsize=14, fontweight='bold', color=color)
                    axs[i, j].axis('off')     

    plt.show()


def sample_phase_images(datasets_dir: Path, output_dir: Path, n_samples: int = 2) -> None:
    """
    Recursively find all .h5/.hdf5 files in datasets_dir, pick n_samples random
    phase frames from each, and save them as .jpeg in output_dir.

    The output filenames encode the source: <h5_stem>_frame<idx>.jpeg
    Files without an Images/Phase dataset are skipped.
    """
    import random

    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(
        list(datasets_dir.rglob('*.h5')) + list(datasets_dir.rglob('*.hdf5'))
    )

    if not h5_files:
        print(f"No .h5 or .hdf5 files found in {datasets_dir}")
        return

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'Images' not in f or 'Phase' not in f['Images']:
                    print(f"Skipping {h5_path.name} (no Images/Phase)")
                    continue

                phase_ds = f['Images']['Phase']

                # Handle both flat dataset (N, H, W) and group-of-frames layouts
                if isinstance(phase_ds, h5py.Dataset):
                    num_frames = phase_ds.shape[0]
                    indices = random.sample(range(num_frames), min(n_samples, num_frames))
                    for idx in indices:
                        img = phase_ds[idx]
                        _save_phase_jpeg(img, output_dir, h5_path, idx)
                elif isinstance(phase_ds, h5py.Group):
                    frame_keys = sorted(phase_ds.keys())
                    chosen = random.sample(frame_keys, min(n_samples, len(frame_keys)))
                    for key in chosen:
                        img = phase_ds[key][:]
                        _save_phase_jpeg(img, output_dir, h5_path, key)

        except Exception as e:
            print(f"Error processing {h5_path.name}: {e}")

    print(f"Saved sample phase images to {output_dir}")


def sample_tiff_images(datasets_dir: Path, output_dir: Path) -> None:
    """
    Recursively find all .tif/.tiff files in datasets_dir, pick a single random
    frame from each, and save it as .jpeg in output_dir.

    Single-page TIFFs save the whole image; multi-page TIFFs (including .ome.tif)
    pick one random page.
    """
    import random

    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(
        list(datasets_dir.rglob('*.tif'))
        + list(datasets_dir.rglob('*.tiff'))
        + list(datasets_dir.rglob('*.ome.tif'))
    )
    # deduplicate (*.tif already catches *.ome.tif)
    tiff_files = sorted(set(tiff_files))

    if not tiff_files:
        print(f"No .tif or .tiff files found in {datasets_dir}")
        return

    for tiff_path in tiff_files:
        try:
            with tifffile.TiffFile(str(tiff_path)) as tif:
                num_pages = len(tif.pages)
                idx = random.randint(0, num_pages - 1)
                img = tif.pages[idx].asarray()

            rel = tiff_path.relative_to(datasets_dir)
            name = rel.with_suffix('').as_posix().replace('/', '_')
            out_path = output_dir / f"{name}_frame{idx}.jpeg"

            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = convert_to_uint8(img)

            pil_img = Image.fromarray(img)
            if pil_img.mode not in ('L', 'RGB'):
                pil_img = pil_img.convert('L')
            pil_img.save(out_path, format='JPEG')

        except Exception as e:
            print(f"Error processing {tiff_path.name}: {e}")

    print(f"Saved {len(tiff_files)} sample tiff images to {output_dir}")


def _save_phase_jpeg(img: np.ndarray, output_dir: Path, h5_path: Path, frame_id) -> None:
    """Save a single phase frame as a JPEG."""
    # Build a unique filename from the relative path of the h5 file
    rel = h5_path.relative_to(h5_path.parents[1])  # keeps subfolder/filename
    name = rel.with_suffix('').as_posix().replace('/', '_')
    out_path = output_dir / f"{name}_frame{frame_id}.jpeg"

    pil_img = Image.fromarray(img)
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')
    pil_img.save(out_path, format='JPEG')


if __name__ == '__main__':
    # repack_hdf5("/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/28_10_5min.h5")
    # repack_hdf5("/home/ubuntu/PhagoPred/PhagoPred/Datasets/ExposureTest/28_10_10min.h5")
    # sample_phase_images("/home/ubuntu/PhagoPred/PhagoPred/Datasets", "/home/ubuntu/PhagoPred/PhagoPred/Datasets/sample_phase_images", n_samples=5)
    sample_tiff_images('/home/ubuntu/thor_server', '/home/ubuntu/PhagoPred/PhagoPred/Datasets/sample_tiff_images')
    # copy_hdf5_groups(Path('PhagoPred') / 'Datasets' / 'ExposureTest' / 'old' / '03_10_2500.h5', 
    #                  Path('PhagoPred')/'Datasets'/'ExposureTest'/ 'old' / '03_10_2500_new.h5',
    #                  ['Images'])
    # print_cell_deaths()
    # hdf5_from_tiffs(Path("D:/27_05_1"), 
    #                 Path('D:/27_05.h5'),
    #                 phase_channel=1,
    #                 epi_channel=2,
    #                 )
    # repack_hdf5()
    # copy_hdf5_groups(Path('PhagoPred') / 'Datasets' / 'ExposureTest' / 'old' / '07_10_0.h5', Path('PhagoPred')/'Datasets'/'ExposureTest'/'07_10_0.h5', ['Images'])
    # copy_hdf5_groups(Path('PhagoPred') / 'Datasets' / 'ExposureTest' / 'old' / '10_10_5000.h5', Path('PhagoPred')/'Datasets'/'ExposureTest'/'10_10_5000.h5', ['Images'])

    # with h5py.File(Path('D:/27_05.h5'), 'r') as f:
    #     dset = f['Images/Epi/Data']
    #     plt.imsave('D:/test0.png', dset[0])
    #     plt.imsave('D:/test200.png', dset[200])
        # print(dset.shape)
        # print(dset[0])
    # get_hist('intensity_mean')
    # plot_correlations(['area', 'intensity_mean', 'intensity_variance', 'perimeter', 'speed', 'displacement_speed', 'phagocyte_density_500', 'perimeter_over_area'])
    # hdf5_from_nd2(r'D:\7_3\260228.nd2', Path('PhagoPred') / 'Datasets' / 'mac_07_03.h5')
    # with h5py.File(SETTINGS.DATASET, 'r+') as f:
    #     del f['Cells']['Phagocytosis']
    #     del f['KMeans']
    #     del f['PCA']
        # f['Images'].attrs['Image size / pixels'] = [1430, 1750]
    # plot_tracks('tracks2.png')
    # make_short_test_copy(Path('PhagoPred') / 'Datasets' / 'mac_07_03.h5', Path('PhagoPred') / 'Datasets' / 'mac_07_03_short.h5')
    # coco_to_semantic_training_data(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'images',
    #                                Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'labels.json',
    #                                Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8' / 'images',
    #                                Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8' / 'masks')
    # rename_files_in_folder(Path('/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/20x_flir'))
    # split_coco_by_category(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'labels_orig.json')
    # split_image_and_coco(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'images_orig', 
    #                      Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'labels_orig.json',
    #                      Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'images', 
    #                      Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_8' / 'labels.json')
    # with open(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '10x_flir_mac_only' / 'model_2' / 'Training_Data' / 'validate' / 'labels.json', 'r') as f:
    #     test_coco = json.load(f)
    # seg_lengths = np.array([len(annotation['segmentation'][0]) for annotation in test_coco['annotations']])
    # print(seg_lengths)
    # print(seg_lengths[seg_lengths <= 4])
    # im_path = Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_mac_only' / 'images' / '0000_00.png'
    # im_show = show_segmentation(plt.imread(im_path), 
    #                   mask_funcs.coco_to_masks(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir_mac_only' / 'labels.json',
    #                                            im_path)[0])
    # plt.imsave(im_path.parent / 'show_am.png', im_show)
    # test_list = [0, 1, 1, 1, 2, 5, 7, 8, 12, 54, 76, 79, 80]
    # print(split_list_into_sequences(test_list))
    # print(split_list_into_sequences(test_list, return_indices=True))
    # make_short_test_copy(orig_file=Path(r"C:\Users\php23rjb\Downloads\temp\mac_test.hdf5"), copy_file=Path('PhagoPred')/'Datasets'/'mac_short.h5')
    # crop_hdf5(Path('PhagoPred')/'Datasets'/'mac_short.h5', crop_size=[1824, 2736])
    #copy_hdf5_groups(SETTINGS.DATASET, Path('PhagoPred')/'Datasets'/'no_filter01_short_temp.h5', ['Images'])
    # create_hdf5(Path('PhagoPred') / 'Datasets' / 'secondwithlight.h5', r'C:\Users\php23rjb\Downloads\second_phase', r'C:\Users\php23rjb\Downloads\second_epi')
    # rename_datasets_in_group('PhagoPred/Datasets/secondwithlight - Copy.h5', '/Images/Phase')
    # array = read_tiff(Path('03') / 't0000_mask.tif')
    # print(np.shape(array))
    # save_tiff(array, Path('03') / 't0000_mask_test.tif')
    # array = read_tiff(Path('03') / 't0000_mask_test.tif')
    # print(np.shape(array))
    # add_empty_epi()
    # for file in (Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8' / 'masks')
