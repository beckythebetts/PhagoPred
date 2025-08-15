import sys
sys.path.insert(0, 'detectron2')

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, inference_context
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import Instances
import torch, detectron2
import torch.nn.functional as F
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
from detectron2.data import transforms as T
from detectron2.data import build_detection_train_loader
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead
from detectron2.structures import Instances
from detectron2.modeling.roi_heads.mask_head import ROI_MASK_HEAD_REGISTRY

from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
import copy

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import logging
import time
import datetime
import json
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from PhagoPred.utils import tools, mask_funcs
from PhagoPred.detectron_segmentation import train
from PhagoPred import SETTINGS


import random

def get_finetuning_dataset(hdf5_file: Path = SETTINGS.DATASET, model_dir: Path = SETTINGS.MASK_RCNN_MODEL):
    """For each cell in death_frames, take frames before and after cell death. 
    Crop images/masks to minimum required to cover cell.
    Split into train/val folders based on Cell Idx (~80/20).
    """
    base_dir = Path(model_dir) / 'Fine_Tuning_Data'
    tools.remake_dir(base_dir)

    # Define directories for train and val
    dirs = {}
    for split in ['train', 'validate']:
        split_dir = base_dir / split
        images_dir = split_dir / 'images'
        tools.remake_dir(images_dir)
        dirs[split] = {
            'root': split_dir,
            'images': images_dir,
            'json_file': split_dir / 'labels.json',
            'coco_json': {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "Macrophage"},
                    {"id": 2, "name": "Dead Macrophage"},
                ]
            }
        }

    # Load death_frames
    death_frames = pd.read_csv(model_dir / 'death_frames.txt', sep='|', skiprows=1, engine='python', dtype=str)
    death_frames = death_frames.iloc[:, 1:3]
    death_frames.columns = ['Cell Idx', 'Death Frame']
    death_frames = death_frames.map(lambda x: x.strip() if isinstance(x, str) else x)
    death_frames = death_frames[death_frames['Death Frame'].str.isnumeric()].astype(int)

    # Split cell ids into train/val
    cell_ids = death_frames['Cell Idx'].unique()
    val_count = max(1, int(len(cell_ids) * 0.2))
    val_cell_ids = set(random.sample(list(cell_ids), val_count))

    frames_sample = np.array([1, 2, 3, 4, 5, 50, 100, 150, 200])

    annotation_id = 1
    image_id = 1

    pbar = tqdm(total=len(death_frames) * len(frames_sample) * 2)

    with h5py.File(hdf5_file, 'r') as f:
        for cell_idx, death_frame in zip(death_frames['Cell Idx'], death_frames['Death Frame']):
            split = 'validate' if cell_idx in val_cell_ids else 'train'
            coco_json = dirs[split]['coco_json']
            images_dir = dirs[split]['images']

            # Alive frames
            for alive_frame in death_frame - frames_sample:
                if alive_frame < 0:
                    pbar.update(1)
                    continue

                mask = f['Segmentations']['Phase'][alive_frame][:] == cell_idx
                if not mask.any():
                    pbar.update(1)
                    continue

                x_coords, y_coords = np.nonzero(mask)
                x_crop = slice(np.min(x_coords), np.max(x_coords) + 1)
                y_crop = slice(np.min(y_coords), np.max(y_coords) + 1)

                image = f['Images']['Phase'][alive_frame]
                image = image[x_crop, y_crop]
                mask = mask[x_crop, y_crop]

                image_file = f'{cell_idx}_{alive_frame}_alive.jpeg'
                plt.imsave(images_dir / image_file, image / 255, cmap='gray')

                mask_funcs.add_coco_annotation(coco_json, image_file, mask, image_id, annotation_id, 1)

                annotation_id += 1
                image_id += 1
                pbar.update(1)

            # Dead frames
            for dead_frame in death_frame + frames_sample:
                if dead_frame > SETTINGS.NUM_FRAMES:
                    pbar.update(1)
                    continue

                mask = f['Segmentations']['Phase'][dead_frame][:] == cell_idx
                if not mask.any():
                    pbar.update(1)
                    continue

                x_coords, y_coords = np.nonzero(mask)
                x_crop = slice(np.min(x_coords), np.max(x_coords) + 1)
                y_crop = slice(np.min(y_coords), np.max(y_coords) + 1)

                image = f['Images']['Phase'][dead_frame][x_crop, y_crop]
                mask = mask[x_crop, y_crop]

                image_file = f'{cell_idx}_{dead_frame}_dead.jpeg'
                seg_found = mask_funcs.add_coco_annotation(coco_json, image_file, mask, image_id, annotation_id, 2)

                if seg_found == 1:
                    plt.imsave(images_dir / image_file, image / 255, cmap='gray')

                annotation_id += 1
                image_id += 1
                pbar.update(1)

    pbar.close()

    # Save COCO jsons
    for split in dirs.keys():
        with open(dirs[split]['json_file'], 'w') as f:
            json.dump(dirs[split]['coco_json'], f)

            
class ClassifierHeadFineTuner(train.MyTrainer):
    """Freeze everything except classification head, and remove all augemtnations in mapper (can't set augmentation sizes to image size like in MyTrainer as image crop sizes vary)."""
    @classmethod
    def build_optimiser(cls, cfg, model):
        """Freezes eveyhting but classificaiton head."""
        for name, param in model.named_parameters():
            if "roi_heads.box_predictor.cls_score" in name:
                param.requires_grad = True

            else:
                param.requires_grad = False
        
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.SOLVER.BASE_LR
        )
    
    def build_train_loader(cls, cfg):
        """Set mapper as no resize mapper,"""
        return build_detection_train_loader(cfg, mapper=no_resize_mapper)

def no_resize_mapper(dataset_dict):
    """
    A custom mapper that loads image and annotations *without* resizing.
    Assumes dataset_dict is in Detectron2 standard format.
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # don't modify original dict

    # Load image from file
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image = torch.as_tensor(image.transpose(2, 0, 1).copy())
    dataset_dict["image"] = image

    # Load annotations (if training)
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, [], image.shape[:2])
            for obj in dataset_dict.pop("annotations")
        ]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])

    return dataset_dict



def train(directory=SETTINGS.MASK_RCNN_MODEL):
    """
    Fine tune the Mask R-CNNmodel on cropped cell images just before and after cell death. Freeze everything expect for classification head.
    """
    # if "DiceLSMaskHead" not in ROI_MASK_HEAD_REGISTRY._obj_map:
    #     ROI_MASK_HEAD_REGISTRY.register(DiceLSMaskHead)
    setup_logger()

    dataset_dir = directory / 'Fine_Tuning_Data'
    config_directory = directory / 'Model'
    register_coco_instances("my_dataset_train", {}, str(dataset_dir / 'train' / 'labels.json'), str(dataset_dir / 'train' / 'images'))
    register_coco_instances("my_dataset_val", {},str(dataset_dir / 'validate' / 'labels.json'), str(dataset_dir / 'validate' / 'images'))

    train_metadata = MetadataCatalog.get("my_dataset_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(str(directory / 'Model' / 'config.yaml'))
    cfg.OUTPUT_DIR = str(directory / 'Model')
    cfg.MODEL.WEIGHTS = str(directory / 'model_final.pth')
    cfg.SOLVER.BASE_LR = 0.0001 # Decrease learnign rate for fine tuning (was 0.00025 for main training)

    trainer = ClassifierHeadFineTuner(cfg)
    trainer.resume_or_load = False
    trainer.train()

    

    plot_loss(config_directory)
    

def plot_loss(cfg_dir):
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines
    
    experiment_metrics = load_json_arr(cfg_dir / 'metrics.json')
    plt.clf()
    plt.rcParams["font.family"] = 'serif'
    plt.scatter([x['iteration'] for x in experiment_metrics if 'total_loss' in x], [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color='navy')
    plt.scatter(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], color='red')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    plt.savefig(cfg_dir / 'loss_plot.png')
    plt.clf()


def main():
    train()
    # get_finetuning_dataset()
    # plot_loss(Path("PhagoPred/detectron_segmentation/models/27_05_mac_finetune/Model"))

if __name__ == '__main__':
    main()