from typing import Union
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import sys
import skimage
import json
import pycocotools
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
import copy
import cv2
# import mahotas
from scipy.ndimage import binary_erosion


from PhagoPred import SETTINGS
from PhagoPred.utils import tools

def add_coco_annotation(coco_json: dict, image_name: str, mask: np.ndarray, image_id: int, annotation_id: int, category_id: int):
    # Get contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6:  # valid polygon must have at least 3 points
            segmentation.append(contour)

    if not segmentation:
        return 0 # skip if no valid contours

    # Compute bbox and area from the binary mask
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    if w < 3 or h < 3:
        return 0
    bbox = [float(x), float(y), float(w), float(h)]
    area = float(np.sum(mask))

    coco_json["images"].append({
        "id": image_id,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "file_name": image_name,
    })

    coco_json["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0
    })
    return 1


def coco_to_masks(coco_file: Union[Path,str], im_name: str) -> dict[str, np.ndarray]:
    """
    Get a mask for each class category in coco_file
    Parameters:
        coco_file: coco file containing annnotations for im_name
        im_name: image for which to get mask/s
    Returns: 
        dictionary {category name: np array mask}
    """
    with open(coco_file, 'r') as f:
        coco_json = json.load(f)

    im_id = None
    for image in coco_json['images']:
        if image['file_name'] == im_name.name:
            im_id = image['id']
            height = image['height']
            width = image['width']
            break

    assert im_id is not None, f"image {im_name} not found"

    masks = {category["id"]: np.zeros(shape=(height, width), dtype=np.int32) for category in coco_json["categories"]}
    category_counts = {category["id"]: 0 for category in coco_json['categories']}

    for annotation in coco_json['annotations']:
        if annotation['image_id'] == im_id:
            category_id = annotation['category_id']
            segmentation = annotation['segmentation']

            # Handle multiple polygons per annotation
            # segmentation can be list of polygons or RLE (not handled here)
            if isinstance(segmentation, list):
                # segmentation is list of polygons, possibly multiple
                instance_mask = np.zeros((height, width), dtype=bool)
                for polygon in segmentation:
                    # polygon is list of x,y coords flat
                    outline_coords = np.array(polygon).reshape(-1, 2)
                    outline_coords = outline_coords[:, [1, 0]]  # swap x,y to y,x for mask indexing
                    poly_mask = skimage.draw.polygon2mask((height, width), outline_coords)
                    instance_mask = np.logical_or(instance_mask, poly_mask)
                
                category_counts[category_id] += 1
                masks[category_id][instance_mask] = category_counts[category_id]

            else:
                # If segmentation is RLE (not handled here), you could add support later
                raise NotImplementedError("RLE segmentation not supported yet")

    id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}
    masks = {id_to_name[id]: mask for id, mask in masks.items()}
    
    return masks

# def coco_to_masks(coco_file: Union[Path,str], im_name: str) -> dict[str, np.ndarray]:
#     """
#     Get a mask for each class category in coco_file
#     Parameters:
#         coco_file: coco file containing annnotations for im_name
#         im_name: image for which to get mask/s
#     Returns: 
#         dictionary {category name: np array mask}
#     """
#     with open(coco_file, 'r') as f:
#         coco_json = json.load(f)

#     im_id = None
#     for image in coco_json['images']:
#         if image['file_name'] == im_name.name:
#             im_id = image['id']
#             height = image['height']
#             width = image['width']
#             break

#     assert im_id is not None, f"image {im_name} not found"

#     masks = {category["id"]: np.zeros(shape=(height, width)) for category in coco_json["categories"]}
#     category_counts = {category["id"]: 0 for category in coco_json['categories']}

#     for annotation in coco_json['annotations']:
#         if annotation['image_id'] == im_id:
            
#             outline_coords = np.array(np.array(annotation['segmentation']).reshape(-1, 2))
#             outline_coords = outline_coords[:, [1,0]]
#             binary_mask = skimage.draw.polygon2mask((height, width), outline_coords)
#             for category_id, mask in masks.items():
#                 if annotation['category_id'] == category_id:
#                     category_counts[category_id] += 1
#                     mask[binary_mask] = category_counts[category_id]
    
#     id_to_name = {cat['id']: cat['name'] for cat in coco_json['categories']}
#     masks = {id_to_name[id]: mask for id, mask in masks.items()}

#     return masks

def boolean_combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    """
    Combine multiple masks (values 1, .., N) to one binary mask
    Parameters:
        masks: list of masks
    Returns:
        boolean mask showing where objects are in any of masks
    """
    binary_mask = np.zeros_like(masks[0])
    for mask in masks:
        binary_mask = np.logical_or(binary_mask, mask > 0)
    
    return binary_mask

def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    """
    Combine multiple masks so that all objects are indexed
    Parameter:
        masks: list of masks
    Returns:
        mask
    """
    combined_mask = np.zeros_like(masks[0])
    idx_count = 0
    for mask in masks:
        combined_mask = np.where(mask>0, mask+idx_count, combined_mask)
        idx_count = np.max(combined_mask)
    
    return combined_mask

import json
from collections import defaultdict

def analyze_coco_annotations(coco_json_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Count total images
    total_images = len(data.get('images', []))

    # Build category ID -> name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}

    # Count instances per category
    category_instance_counts = defaultdict(int)
    for ann in data.get('annotations', []):
        cat_id = ann['category_id']
        category_instance_counts[cat_id] += 1

    # Print results
    print(f"Total images: {total_images}\n")
    print("Instances per category:")
    for cat_id, count in sorted(category_instance_counts.items(), key=lambda x: x[1], reverse=True):
        cat_name = category_id_to_name.get(cat_id, f"Unknown (ID {cat_id})")
        print(f"  {cat_name}: {count}")

# if __name__ == "__main__":
#     path_to_coco_json = "path/to/your/coco_annotations.json"  # ← Replace this
#     analyze_coco_annotations(path_to_coco_json)

# def coco_to_masks(coco_file, im_name):
#     with open(coco_file, 'r') as f:
#         coco_json = json.load(f)

#     im_id = None
#     for image in coco_json['images']:
#         if image['file_name'] == im_name.name:
#             im_id = image['id']
#             height = image['height']
#             width = image['width']
#             break

#     assert im_id is not None, f"image {im_name} not found"

#     cell_mask = np.zeros(shape=(height, width))
#     cluster_mask = np.zeros(shape=(height, width))

#     for annotation in coco_json['annotations']:
#         if annotation['image_id'] == im_id:
#             outline_coords = np.array(np.array(annotation['segmentation']).reshape(-1, 2))
#             outline_coords = outline_coords[:, [1,0]]
#             binary_mask = skimage.draw.polygon2mask(cell_mask.shape, outline_coords)
#             if annotation['category_id'] == 1:
#                 cell_mask[binary_mask] = annotation['id']
#             elif annotation['category_id'] == 2:
#                 cluster_mask[binary_mask] = annotation['id']
#     return cell_mask, cluster_mask

def coco_to_binary_mask(coco_file, im_name):
    with open(coco_file, 'r') as f:
        coco_json = json.load(f)

    im_id = None
    for image in coco_json['images']:
        if image['file_name'] == im_name.name:
            im_id = image['id']
            height = image['height']
            width = image['width']
            break

    assert im_id is not None, f"image {im_name} not found"

    binary_mask = np.zeros(shape=(height, width))

    for annotation in coco_json['annotations']:
        if annotation['image_id'] == im_id:
            outline_coords = np.array(np.array(annotation['segmentation']).reshape(-1, 2))
            outline_coords = outline_coords[:, [1,0]]
            mask = skimage.draw.polygon2mask(binary_mask.shape, outline_coords)
            binary_mask[mask] = 1
    
    return binary_mask

# def get_areas(mask):
#     idxs = tools.unique_nonzero(mask)
#     expanded_mask = (np.expand_dims(mask, 2) == np.expand_dims(idxs, (0,1)))
#     return np.sum(expanded_mask, axis=(0,1))

def get_areas(mask):
    _, counts = np.unique(mask, return_counts=True)
    counts[0] = 0
    return counts

def get_perimeters(mask, cell_batchsize=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device='cpu'
    # idxs = np.unique(mask[mask>0])
    # mask = (np.expand_dims(mask, 2) == np.expand_dims(idxs, (0,1)))
    # mask = torch.tensor(mask).to(device)
    # print(mask.shape, torch.unique(mask))

    mask = torch.tensor(mask.astype(np.int16)).to(device)
    # idxs = torch.unique(mask[mask>0])
    num_cells = int(torch.max(mask)+1)
    perimeters = torch.empty(size=(num_cells,))
    for first_cell in range(1, num_cells, cell_batchsize):
        last_cell = min(first_cell+cell_batchsize, num_cells)
        idxs = torch.arange(first_cell, last_cell).to(device)
        expanded_mask = mask.unsqueeze(0) == idxs.unsqueeze(1).unsqueeze(2)
        kernel = torch.tensor([[1, 1, 1],
                            [1, 9, 1],
                            [1, 1, 1]]).to(device)
        # print(mask.shape, kernel.shape)
        padded_masks = torch.nn.functional.pad(expanded_mask, (1, 1, 1, 1), mode='constant', value=0)
        # print(padded_masks.shape)
        conv_result = torch.nn.functional.conv2d(padded_masks.squeeze().unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                    padding=0).squeeze()
        perimeters[first_cell:last_cell] = torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float()
    perimeters[0] = 0
    # perimeters = np.insert(perimeters.cpu().numpy(), 0, 0)
    perimeters = perimeters.cpu().numpy()

    return perimeters

def squash_idxs(array):
    """
    Reindex all cells in an image to desne integers
    """
    _, inverse = np.unique(array, return_inverse=True)
    return inverse

def get_perimeters_over_areas(mask):
    return np.insert(get_perimeters(mask)[1:] / get_areas(mask)[1:], 0, 0)

def get_densities(mask, search_radius=500):
    centres = np.array(get_centres(mask))
    distances = np.linalg.norm(np.expand_dims(centres, 1) - np.expand_dims(centres, 0), axis=(2))
    densities = np.sum(distances <= search_radius, axis=0) - 1
    return densities
    

def get_centres(mask, x_mesh_grid=None, y_mesh_grid=None):
    """
    mask, np array shape [x, y]
    """
    if x_mesh_grid is None or y_mesh_grid is None:
        x_mesh_grid, y_mesh_grid = np.meshgrid(np.arange(mask.shape[1]), np.arange(mask.shape[0]))
    # areas = get_areas(mask)
    centres = [np.array(get_centre(mask==idx, x_mesh_grid, y_mesh_grid)) for idx in np.unique(mask)]
    return centres


    
def get_centre(mask, x_mesh_grid=None, y_mesh_grid=None):
    if x_mesh_grid is None or y_mesh_grid is None:
        x_mesh_grid, y_mesh_grid = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]))
    area = np.sum(mask)
    x_centre = np.sum(mask*x_mesh_grid)/area
    y_centre = np.sum(mask*y_mesh_grid)/area
    return (x_centre, y_centre)


def dist_between_points(point_1, point_2):
    print(point_1, point_2)
    return np.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2)


def to_instance_mask(mask):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    separated_cells = np.zeros_like(mask)
    for label in range(1, num_labels):
        separated_cells[labels == label] = label
    return separated_cells


def to_masks(image_path, type):
    mask_vals = {"amoeba": 127, "yeast": 254, "proximity": 255}
    seg_mask = cv2.imread(image_path)
    seg_mask = seg_mask[:, 1024:]
    if type == 'proximity':
        return np.where(seg_mask[:, :, 2] == 255, 1, 0)
    else:
        return to_instance_mask(np.where(seg_mask[:, :, 2] == mask_vals[type], 1, 0))


class SplitMask:
    def __init__(self, mask_full):
        self.mask_full = mask_full
        self.i = 0
        self.max = torch.max(self.mask_full)
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        while self.i not in self.mask_full and self.i <= self.max:
            self.i += 1
        if self.i <= self.max:
            return torch.where(self.mask_full == self.i, 1, 0), self.i
        else:
            raise StopIteration


def split_mask(mask_full, use_torch=False, return_indices=False):
    # Return indices=True only works if use_troch=True
    if use_torch:
        #masks = torch.tensor([torch.where(mask_full == i + 1, 1, 0) for i in range(0, int(torch.max(mask_full))) if i + 1 in mask_full])
        masks = []
        max_val = int(torch.max(mask_full))
        masks_dict = {}
        for i in range(1, max_val + 1):
            if i in mask_full:
                mask = torch.where(mask_full == i, 1, 0)
                if return_indices:
                    masks_dict[i] = mask
                else:
                    masks.append(mask)
        if not return_indices:
            masks = torch.stack(masks)
    else:
        masks = [[np.where(mask_full == i + 1, 1, 0)] for i in range(0, np.max(mask_full)) if i + 1 in mask_full]
    if return_indices:
        return masks_dict
    else:
        return masks


def circle_equ(x, y, centre, radius):
    return (x-centre[0])**2 + (y-centre[1])**2 <= radius**2


def create_circle(centre, radius, array_shape=(1024, 1024)):
    circle = circle_equ(np.arange(0, array_shape[0], 1), np.arange(0, array_shape[1], 1)[:, np.newaxis], centre, radius)
    return circle


def torch_circle(centre, radius, array_shape=SETTINGS.IMAGE_SIZE):
    circle = circle_equ(torch.arange(0, array_shape[0], 1).cuda().unsqueeze(1), torch.arange(0, array_shape[1], 1).cuda().unsqueeze(0), centre, radius)
    return circle


def cal_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    result = np.sum(intersection) / np.sum(union)
    return result


# def mask_outline(mask, thickness=3):
#     expanded_mask = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=2*thickness+1, stride=1, padding=thickness) > 0
#     outline = (expanded_mask.byte().squeeze() - mask).bool()
#     return outline

def mask_outline(mask, thickness=3):
    expanded_mask = (F.max_pool2d(mask.float().unsqueeze(0), kernel_size=2*thickness+1, stride=1, padding=thickness) > 0)
    # outline = (expanded_mask.byte().squeeze() - mask).bool()
    outline = (expanded_mask.byte().squeeze() - mask.byte()).bool()

    return outline


def mask_outlines(mask, thickness=3):
    expanded_mask = F.max_pool2d(mask.float().unsqueeze(0), kernel_size=2*thickness+1, stride=1, padding=thickness).to(torch.int16)
    outlines = expanded_mask.squeeze() - mask
    return outlines

def combine_cells_clusters(cell_mask_orig, cluster_mask):
    cell_mask = copy.deepcopy(cell_mask_orig)
    cell_mask[cell_mask>0] += 1
    cell_mask[cluster_mask>0] = 1
    return cell_mask

# def find_centre(mask):
#     coords = torch.nonzero(mask)
#     len = coords.shape[0]
#     x_mean = torch.sum(coords[:, 1]) / len
#     y_mean = torch.sum(coords[:, 0]) / len
#     return x_mean, y_mean
def get_crop_indices(center, side_length, image_size):
    """
    Get the crop indices for a square crop from an image.

    Parameters:
        center (tuple): (y, x) coordinates of the center of the crop.
        side_length (int): Length of the sides of the square crop.
        image_size (tuple): (height, width) of the original image.

    Returns:
        tuple: (y_start, y_end, x_start, x_end) indices for cropping.
    """

    # Calculate half the side length
    half_length = side_length // 2

    # Determine the crop boundaries
    y_center, x_center = center
    y_start = y_center - half_length
    y_end = y_center + half_length
    x_start = x_center - half_length
    x_end = x_center + half_length

    # Adjust boundaries if they exceed image dimensions
    y_start = max(0, y_start)
    y_end = min(image_size[0], y_end)
    x_start = max(0, x_start)
    x_end = min(image_size[1], x_end)

    # Adjust the center if the crop is adjusted
    if y_end - y_start < side_length:
        if y_start == 0:
            y_end = y_start + side_length
        else:
            y_start = y_end - side_length

    if x_end - x_start < side_length:
        if x_start == 0:
            x_end = x_start + side_length
        else:
            x_start = x_end - side_length

    # Ensure the crop dimensions are valid
    y_start = int(max(0, y_start))
    y_end = int(min(image_size[0], y_end))
    x_start = int(max(0, x_start))
    x_end = int(min(image_size[1], x_end))

    return (y_start, y_end, x_start, x_end)


def get_crop_indices_all(centers, side_length, image_size):
    """
    Get the crop indices for square crops from an image for multiple centers.

    Parameters:
        centers (np.array): Array of shape (N, 2) containing (y, x) coordinates of the crop centers.
        side_length (int): Length of the sides of the square crop.
        image_size (tuple): (height, width) of the original image.

    Returns:
        np.array: Array of shape (N, 4) containing (y_start, y_end, x_start, x_end) for each crop.
    """
    # Calculate half the side length
    half_length = side_length // 2

    # Unpack image dimensions
    width, height = image_size

    # Compute initial crop boundaries
    y_centers, x_centers = centers[:, 0], centers[:, 1]
    y_start = y_centers - half_length
    y_end = y_centers + half_length
    x_start = x_centers - half_length
    x_end = x_centers + half_length

    # Clip boundaries to the image dimensions
    y_start = np.clip(y_start, 0, height)
    y_end = np.clip(y_end, 0, height)
    x_start = np.clip(x_start, 0, width)
    x_end = np.clip(x_end, 0, width)

    # Adjust for cases where the crop is smaller than the side length
    adjusted_y_end = np.where((y_end - y_start) < side_length, y_start + side_length, y_end)
    adjusted_y_start = np.where((y_end - y_start) < side_length, y_end - side_length, y_start)
    adjusted_x_end = np.where((x_end - x_start) < side_length, x_start + side_length, x_end)
    adjusted_x_start = np.where((x_end - x_start) < side_length, x_end - side_length, x_start)

    # Ensure final boundaries stay within image dimensions
    y_start = np.clip(adjusted_y_start, 0, height)
    y_end = np.clip(adjusted_y_end, 0, height)
    x_start = np.clip(adjusted_x_start, 0, width)
    x_end = np.clip(adjusted_x_end, 0, width)
    
    # Stack results into a single array
    crop_indices = np.stack((y_start, y_end, x_start, x_end), axis=-1)
    crop_indices[np.isnan(crop_indices)] = 0
    crop_indices = crop_indices.astype(int)

    
    return crop_indices

# def get_border_representation(mask_im, num_cells, f, frame):
#     """
#     Return list of cell contours from mask. 
#     [cell_idx: ](Nx2) array of (x, y) coords]
#     num_cells must be totalnumber of cells in datset (for indexing to be correct)
#     """
#     expanded_mask = (np.expand_dims(mask_im, 0) == np.expand_dims(np.arange(num_cells), (1,2)))
#     crop_idxs = get_crop_indices_all(np.stack((tools.get_features_ds(f['Cells']['Phase'], 'x')[frame], 
#                                                             tools.get_features_ds(f['Cells']['Phase'], 'y')[frame]), axis=1), 150, SETTINGS.IMAGE_SIZE)
    
#     x_idxs = np.array([np.arange(xmin, xmax) if xmin != xmax else np.zeros(150) for xmin, xmax in zip(crop_idxs[:,2], crop_idxs[:,3])]).astype(int)[:, :, np.newaxis]
#     y_idxs = np.array([np.arange(ymin, ymax) if ymin != ymax else np.zeros(150) for ymin, ymax in zip(crop_idxs[:,0], crop_idxs[:,1])]).astype(int)[:, np.newaxis, :]

#     cropped_masks = expanded_mask[np.arange(num_cells)[:, None, None], x_idxs, y_idxs]
#     cell_contours = [skimage.measure.find_contours(mask, level=0.5, fully_connected='high') for mask in cropped_masks]
#     cell_contours = [cell_contour[0] if len(cell_contour)>0 else np.array([]) for cell_contour in cell_contours]
#     for i, cell_contour in enumerate(cell_contours):
#         if len(cell_contour > 0):
#             if (cell_contour[-1] != cell_contour[0]).any():
#                 cell_contours[i] = np.append(cell_contour, cell_contour[0][np.newaxis, :], axis=0)

#     return cell_contours

def get_border_representation(expanded_mask):
    crop_idxs = [get_minimum_mask_crop(mask) for mask in expanded_mask]
    cropped_masks = [mask[crop_idx[0], crop_idx[1]] for mask, crop_idx in zip(expanded_mask, crop_idxs)]

    cell_contours = []
    for mask in cropped_masks:
        try:
            cell_contour = skimage.measure.find_contours(mask, level=0.5, fully_connected='high')
        except ValueError:
            cell_contour = []
        
        cell_contours.append(cell_contour)

    # cell_contours = [skimage.measure.find_contours(mask, level=0.5, fully_connected='high') for mask in cropped_masks]
    cell_contours = [cell_contour[0] if len(cell_contour)>0 else np.array([]) for cell_contour in cell_contours]
    for i, cell_contour in enumerate(cell_contours):
        if len(cell_contour > 0):
            if (cell_contour[-1] != cell_contour[0]).any():
                cell_contours[i] = np.append(cell_contour, cell_contour[0][np.newaxis, :], axis=0)

    return cell_contours

def get_minimum_mask_crop(mask):
    """binary mask of one cell"""
    if mask.any() > 0:
        rows, cols = np.where(mask > 0)
        return slice(rows.min(), rows.max()+1), slice(cols.min(), cols.max()+1)
    
    else:
        # print('NO MASK?')
        return slice(0,0), slice(0,0)

def get_haralick_texture_features(image, mask, distances=[1,3,5, 10, 20], erode_mask=None):
    """mask - binary mask for one cell
    average over all (4) directions
    erode mask to remove perimeter textures from calculation, value is approx num pixels to to remove"""

    # features = np.empty((len(distances), 13))
    features = np.full((len(distances), 13), np.nan)

    if mask is not None:
        if erode_mask is not None:
            struct = np.ones((2*erode_mask+1, 2*erode_mask+1))  
            mask = binary_erosion(mask, structure=struct)

        try:
            row_slice, col_slice = get_minimum_mask_crop(mask)
            image, mask = image[row_slice, col_slice], mask[row_slice, col_slice]
            image = image * mask
        except ValueError:
            return features

    for i, distance in enumerate(distances):
        try:
            features[i] = mahotas.features.haralick(image, distance=distance, ignore_zeros=True, return_mean=True)
        except ValueError as e:
            print(e)
            pass
    return features

def convert_coco_file(coco_file, im_dir, new_folder):
    for im in im_dir.iterdir():
        mask = coco_to_masks(coco_file, im)[0].astype(np.uint16)
        mask = Image.fromarray(mask)
        mask.save(new_folder / f'{im.stem}mask.png')
        shutil.copy(im, new_folder / f'{im.stem}im.png')

import numpy as np
import cv2

def calculate_mean_diameter(training_folder):
    diameters = []
    masks = [plt.imread(mask) for mask in training_folder.glob('*mask*')]
    for img in masks:
        # Iterate over each mask to calculate cell diameters
        for label in np.unique(img):
            if label == 0:  # Skip background (assuming background is labeled 0)
                continue
            
            # Create a binary mask for the current cell
            cell_mask = (img == label).astype(np.uint8)
            
            # Find contours of the cell in the mask
            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate the minimum enclosing circle for the contour
            for contour in contours:
                if len(contour) >= 5:  # Minimum number of points for an ellipse fitting
                    # Fit an ellipse to the contour
                    ellipse = cv2.fitEllipse(contour)
                    # Ellipse parameters: (center, axes, rotation)
                    # The axes of the ellipse correspond to the minor and major diameters
                    diameters.append(np.mean(ellipse[1]))  # Take the average of the axes as the cell diameter
    
    # Return the average diameter of all cells
    return np.mean(diameters).astype(float)  # Fallback to a default if no cells are found

def erode_mask(mask: np.array, erosion_val: int=10) -> np.array:
    struct = np.ones((2*erosion_val+1, 2*erosion_val+1))
    return binary_erosion(mask, structure=struct)

# def clean_coco_json(coco_json_path: Path, image_dir: Path) -> None:
#     with open(coco_json_path, 'r') as f:
#         coco = json.load(f)
    
#     existing_ims = set([im.name for im in image_dir.iterdir()])
#     valid_ims = [img for img in coco['images'] if img['file_name'] in existing_ims]
#     valid_im_ids = set(img['id'] for img in valid_ims)
#     valid_annotations = [ann for ann in coco['annotations'] if ann['image_id'] in valid_im_ids]

#     coco['images'] = valid_ims
#     coco['annotations'] = valid_annotations

#     with open(coco_json_path, 'w') as f:
#         json.dump(coco, f)

def clean_coco_json(coco_json_path: Path, image_dir: Path) -> None:
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    existing_ims = [im for im in image_dir.iterdir() if im.suffix.lower() in image_exts]
    existing_im_names = set(im.name.lower() for im in existing_ims)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    # Filter valid images already in the JSON
    valid_ims = [img for img in images if img['file_name'].lower() in existing_im_names]
    valid_im_ids = set(img['id'] for img in valid_ims)
    valid_annotations = [ann for ann in annotations if ann['image_id'] in valid_im_ids]

    # Track existing file_names in JSON (case-insensitive)
    json_filenames = set(img['file_name'].lower() for img in valid_ims)

    # Determine the next available image ID
    used_ids = {img['id'] for img in images}
    next_id = max(used_ids, default=0) + 1

    # Add missing image files from directory
    new_images = []
    for im in existing_ims:
        if im.name.lower() not in json_filenames:
            try:
                with Image.open(im) as img:
                    width, height = img.size
                new_images.append({
                    "id": next_id,
                    "file_name": im.name,
                    "width": width,
                    "height": height
                })
                next_id += 1
            except Exception as e:
                print(f"Warning: Failed to read image {im.name}: {e}")

    # Update COCO JSON
    coco['images'] = valid_ims + new_images
    coco['annotations'] = valid_annotations

    with open(coco_json_path, 'w') as f:
        json.dump(coco, f)

    print(f"Cleaned COCO JSON:")
    print(f"  - Retained {len(valid_ims)} existing valid images")
    print(f"  - Added {len(new_images)} new image(s) from folder")
    print(f"  - Retained {len(valid_annotations)} annotations")


if __name__ == '__main__':
    # get_centre(np.zeros((5, 5)))
    # array_1 = np.array([[0,0], [1,0], [2, 2]])
    # array_2 = np.array([[0,1], [2, 2], [3, 4], [2, 1]])
    # distances = np.linalg.norm(array_1[:, np.newaxis] - array_2[np.newaxis], axis=2)
    # print(distances)
    # convert_coco_file(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir' / 'labels.json',
    #                   Path('PhagoPred') / 'detectron_segmentation' / 'models' / '20x_flir' / 'images',
    #                   Path('PhagoPred') / 'cellpose_segmentation' / 'Models' / '20x_flir' / 'all')
    # clean_coco_json(
    #     Path("/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/27_05_mac/Fine_Tune_data_full/labels.json"),
    #     Path("/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/27_05_mac/Fine_Tune_data_full/images")
    # )
    analyze_coco_annotations(
        "/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/27_05_mac/Fine_Tune_data_full/labels.json"
    )
    # plt.imsave('/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/20x_flir/mask0003.png', coco_to_masks(Path('/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/20x_flir/labels.json'), Path('0003.png'))[0].astype(np.uint16), cmap='gray') 
