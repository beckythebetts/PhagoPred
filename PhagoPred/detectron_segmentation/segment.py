import sys
from typing import Union
sys.path.insert(0, 'detectron2')

import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

import tifffile as tiff
import numpy as np
import os, json, cv2, random
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import sys
import glob
import h5py
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from detectron2.engine.defaults import DefaultPredictor
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.data import MetadataCatalog, detection_utils as utils
# import torch

class NoResizePredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions

        

def get_model(cfg_dir: Path) -> tuple[dict, detectron2.config.CfgNode]:
    """
    Get training metadata and model from cfg_dir
    Parameters:
        cfg_dir: path containing train_metadata.json, config.yaml. model_final.pth
    Returns:
        train_metadata dict
        cfg
    """
    with open(str(cfg_dir / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_dir / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(cfg_dir / 'model_final.pth')

    return train_metadata, cfg

def get_predictor(cfg) -> tuple[object, Union[DefaultPredictor, NoResizePredictor]]:
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = device_str
    return cfg, NoResizePredictor(cfg)

def seg_image(cfg_dir: Path, 
              im: np.ndarray, 
              train_metadata = None,
              cfg = None,
              predictor = None
            #   categories: tuple[str, ...]
              ) -> dict[str, np.ndarray]:
    """
    Segment a single image using model in cfg_dir
    Parameters:
        cfg_dir: Path to cfg dir
        im: np.ndarry image [X, Y, 3]
        # categories: tuple of the category names to segment
    Returns:
        masks: Dict of {category name: mask}. Each mask shape [X, Y], 0 for background with each item labelled 1, ..,N
    """
    if train_metadata is None or cfg is None:
        train_metadata, cfg = get_model(cfg_dir)
    if predictor is None:
        cfg, predictor = get_predictor(cfg)
    # device = torch.device(device_str)
    # train_metadata, cfg = get_model(cfg_dir)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    # cfg.MODEL.DEVICE = device_str
    # # predictor = DefaultPredictor(cfg)
    # predictor = NoResizePredictor(cfg)
    detectron_outputs = predictor(im)

    if len(detectron_outputs["instances"]) == 0:
        return None

    categories = train_metadata["thing_classes"]

    masks = {category: torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                device=device) for category in categories}
    category_count = {category: 0 for category in categories}
    for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
        class_name = train_metadata['thing_classes'][pred_class]
        instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)

        # ******* ADJUST FOR NON SQUARE IMAGES*********
        # if SETTINGS.REMOVE_EDGE_CELLS:
        #     if torch.any(torch.nonzero(instance_mask)==1) or torch.any(torch.nonzero(instance_mask)==SETTINGS.IMAGE_SIZE[0]-1):
        #         continue

        for category in categories:
            if class_name == category:
                category_count[category] += 1
                masks[category] = torch.where(instance_mask,
                                torch.tensor(category_count[category], dtype=torch.int16),
                                masks[category])
        torch.cuda.empty_cache()

    return {category: mask.cpu().numpy().astype(np.int16) for category, mask in masks.items()}

def seg_dataset(cfg_dir: Path = SETTINGS.MASK_RCNN_MODEL / 'Model', 
                dataset: Path = SETTINGS.DATASET,
                channel: str = 'Phase') -> None:
    
    """
    Create segmentation datset and cell datset in hdf5 file. 
    Add indexed cell masks to segmentation dataset and cell_type as feature in cell dataset.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    train_metadata, cfg = get_model(cfg_dir)

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = device_str
    predictor = DefaultPredictor(cfg)

    categories = train_metadata["thing_classes"]    

    with h5py.File(dataset, 'r+') as f:

        for group_name in ('Segmentations', 'Cells'):
            group = f.require_group(group_name)
            if channel in group:
                del group[channel]

        images_ds = f['Images'][channel]
        segmentations_ds = f.create_dataset(f'Segmentations/{channel}', shape=images_ds.shape,
                                            maxshape=images_ds.shape, dtype='i2')
        cells_group = f.require_group(f'Cells/{channel}')

        for category in categories:
            cells_group.require_dataset(category, 
                                    shape=(images_ds.shape[0], 0),
                                    maxshape=(images_ds.shape[0], None),
                                    dtype=np.float32,
                                    exact=True,
                                    fillvalue=np.nan)

        for frame_idx in tqdm(range(images_ds.shape[0])):
            # sys.stdout.write(f'\rSegmenting image {int(frame_idx)+1} / {f["Images"].attrs["Number of frames"]}')
            # sys.stdout.flush()

            image = images_ds[frame_idx]

            detectron_outputs = predictor(np.stack([np.array(image)]*3, axis=-1))

            # mask = torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
            #                         device=device)
            mask = torch.full_like(detectron_outputs["instances"].pred_masks[0], -1, dtype=torch.int16,
                                   device=device)
            # resize dataset if necassary
            num_instances = len(detectron_outputs['instances'].pred_classes)
            for category in categories:
                current_max_instances = cells_group[category].shape[1]-1
                if num_instances > current_max_instances:
                    cells_group[category].resize(num_instances, axis=1)

            for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
                class_name = train_metadata['thing_classes'][pred_class]
                instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)

                # if SETTINGS.REMOVE_EDGE_CELLS:
                #     if instance_mask[0, :].any() or instance_mask[-1, :].any() or instance_mask[:, 0].any() or instance_mask[:, -1].any():
                #         continue

                
                mask = torch.where(instance_mask, i, mask)
                cells_group[class_name][frame_idx, i] = 1
                # print(torch.unique(mask))
                # cells_ds[frame_idx, i+1, class_name] = 1

            segmentations_ds[frame_idx] = mask.cpu().numpy()
        
def main():
    seg_dataset()

if __name__ == '__main__':
    main()

