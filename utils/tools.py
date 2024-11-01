from pathlib import Path
import shutil
import torch
from PIL import Image
import numpy as np
import imageio

from PhagoPred import SETTINGS
import mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_segmentation(image_array, mask_array, true_mask_array=None):
    image_array, mask_array = torch.tensor(np.stack((image_array, image_array, image_array), axis=-1)), torch.tensor(mask_array)
    outline = mask_funcs.mask_outline(torch.where(mask_array>0, 1, 0), thickness=2)
    image_array[:,:,0][outline] = torch.max(image_array)
    if true_mask_array is not None:
        true_mask_array = torch.tensor(true_mask_array)
        true_outline = mask_funcs.mask_outline(torch.where(true_mask_array>0, 1, 0), thickness=2)
        image_array[:, :, 1][true_outline] = torch.max(image_array)
        #print(image_array.shape, image_array.type)
    return image_array.cpu().numpy()


def remake_dir(path):
    if path.is_dir():
        check = input(f'Delete directory {str(path)}? [y,n] ')
        if check.lower() != 'y':
            raise SystemExit(0)
        shutil.rmtree(path)
    path.mkdir(parents=True)


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


if __name__ == '__main__':
    test_list = [0, 1, 1, 1, 2, 5, 7, 8, 12, 54, 76, 79, 80]
    print(split_list_into_sequences(test_list))
    print(split_list_into_sequences(test_list, return_indices=True))

    # array = read_tiff(Path('03') / 't0000_mask.tif')
    # print(np.shape(array))
    # save_tiff(array, Path('03') / 't0000_mask_test.tif')
    # array = read_tiff(Path('03') / 't0000_mask_test.tif')
    # print(np.shape(array))