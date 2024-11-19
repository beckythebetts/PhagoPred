from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np
import imageio
import h5py
import os

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_segmentation(image_array, mask_array, true_mask_array=None):
    image_array, mask_array = torch.tensor(np.stack((image_array, image_array, image_array), axis=-1)), torch.tensor(mask_array)
    outline = mask_funcs.mask_outline(torch.where(mask_array>0, 1, 0), thickness=2)
    image_array[:,:,0][outline] = torch.max(image_array)
    if true_mask_array is not None:
        true_mask_array = torch.tensor(true_mask_array)
        true_outline = mask_funcs.mask_outline(torch.where(true_mask_array>0, 1, 0), thickness=2)
        image_array[:, :, 1][true_outline] = torch.max(image_array)
    return image_array.cpu().numpy()

def repack_hdf5(file=SETTINGS.DATASET):
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
            # Copy all attributes of the group
            for attr_name, attr_value in src_group.attrs.items():
                dest_group.attrs[attr_name] = attr_value
            
            # Recursively copy datasets and subgroups
            for name, item in src_group.items():
                if isinstance(item, h5py.Dataset):  # Copy datasets directly
                    dest_group.create_dataset(name, data=item[()])
                    # Copy dataset attributes
                    for attr_name, attr_value in item.attrs.items():
                        dest_group[name].attrs[attr_name] = attr_value
                elif isinstance(item, h5py.Group):  # Recurse into subgroups
                    subgroup = dest_group.create_group(name)
                    copy_group(item, subgroup)
        
        # Copy each specified group
        for group_name in groups_to_copy:
            if group_name in src_file:
                src_group = src_file[group_name]
                dest_group = dest_file.create_group(group_name)
                copy_group(src_group, dest_group)
            else:
                print(f"Warning: Group '{group_name}' not found in the source file.")

def make_short_test_copy(orig_file, copy_file, frames=50):
    with h5py.File(orig_file, 'r') as orig:
        with h5py.File(copy_file, 'x') as copy:
            Images = copy.create_group('Images')
            for attr_name, attr_value in orig['Images'].attrs.items():
                Images.attrs[attr_name] = attr_value
            Images.attrs['Number of frames'] = frames
            phase = Images.create_group('Phase')
            epi = Images.create_group('Epi')
            for name, phase_image, epi_image in zip(list(orig['Images']['Phase'].keys())[:frames],
                                                    list(orig['Images']['Phase'].values())[:frames],
                                                    list(orig['Images']['Epi'].values())[:frames]):
                phase.create_dataset(name, data=phase_image[:])
                epi.create_dataset(name, data=epi_image[:])

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

    array[x.round().to(int), y.round().to(int), :] = colour
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


# def append_hdf5(f, array):
#     """
#     Appends a NumPy array to an existing HDF5 dataset.
    
#     Parameters:
#     - f: dataset
#     - array: The array to append to the dataset.
#     """
#     # Resize the dataset to accommodate the new data
#     current_size = f.shape[0]
#     new_size = current_size + array.shape[0]
    
#     # Resize the dataset
#     f.resize((new_size, *f.shape[1:]))
    
#     # Append the new data
#     f[current_size:new_size] = array


if __name__ == '__main__':
    # test_list = [0, 1, 1, 1, 2, 5, 7, 8, 12, 54, 76, 79, 80]
    # print(split_list_into_sequences(test_list))
    # print(split_list_into_sequences(test_list, return_indices=True))
    make_short_test_copy(orig_file=Path('PhagoPred')/'Datasets'/'filter01.h5', copy_file=Path('PhagoPred')/'Datasets'/'filter01_short.h5')
    # array = read_tiff(Path('03') / 't0000_mask.tif')
    # print(np.shape(array))
    # save_tiff(array, Path('03') / 't0000_mask_test.tif')
    # array = read_tiff(Path('03') / 't0000_mask_test.tif')
    # print(np.shape(array))