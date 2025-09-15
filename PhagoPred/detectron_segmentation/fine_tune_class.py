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
from detectron2.modeling.roi_heads import StandardROIHeads
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
import torch
import torch.nn.functional as F

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import cat
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from PhagoPred.utils import tools, mask_funcs
from PhagoPred.detectron_segmentation import train
from PhagoPred import SETTINGS


import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_finetuning_dataset(
        hdf5_file: Path = SETTINGS.DATASET, 
        model_dir: Path = SETTINGS.MASK_RCNN_MODEL,
        include_no_cell: bool = True,
        include_wrong_segs: bool = True,
        padding: int = 20,
        ):
    """For each cell in death_frames, take frames before and after cell death. 
    Crop images/masks to minimum required to cover cell.
    Split into train/val folders based on Cell Idx (~80:20).
    If include wrong_segs, take images labelled as 'Wrong Segmentation' in death_frames.txt and include as no cell (random frames).
    If include_no_cell is true, take random crops of areas with no cell.
    Args
    ----
        Padding: Number of pixels around each cell outline to 
    """
    base_dir = Path(model_dir) / 'Fine_Tuning_Data'
    tools.remake_dir(base_dir)

    crop_shapes = {'train': [], 'validate': []}

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
    death_frames_df = pd.read_csv(model_dir / 'death_frames.txt', sep='|', skiprows=1, engine='python', dtype=str)
    death_frames_df = death_frames_df.iloc[:, 1:3]
    death_frames_df.columns = ['Cell Idx', 'Death Frame']
    death_frames_df = death_frames_df.map(lambda x: x.strip() if isinstance(x, str) else x)

    death_frames = death_frames_df[death_frames_df['Death Frame'].str.isnumeric()].astype(int)
    wrong_segmentations = death_frames_df[death_frames_df['Death Frame'] == 'Wrong Segmentation']
    wrong_segmentations['Cell Idx'] = wrong_segmentations['Cell Idx'].astype(int)

    # Split cell ids into train/val
    cell_ids = death_frames['Cell Idx'].unique()
    val_count = max(1, int(len(cell_ids) * 0.2))
    val_cell_ids = set(random.sample(list(cell_ids), val_count))

    alive_frames_sample = np.array([1, 2, 3, 4, 5, 50, 100, 150, 200])
    dead_frames_sample = np.array([1, 2, 3, 4,5 , 6, 7, 8, 9])
    annotation_id = 1
    image_id = 1

    pbar = tqdm(total=(len(death_frames) * (len(alive_frames_sample) + len(dead_frames_sample)) * 1.5) + len(wrong_segmentations))

    def crop_image_and_mask(image: np.ndarray, mask:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_coords, y_coords = np.nonzero(mask)
        crop_coords = [np.min(x_coords) - padding,
                        np.max(x_coords) + 1 + padding,
                        np.min(y_coords) - padding,
                        np.max(y_coords) + 1 + padding,
                        ]
        crop_coords = np.clip(crop_coords, 0, mask.shape[0]) # Assuming square images
        x_crop = slice(crop_coords[0], crop_coords[1])
        y_crop = slice(crop_coords[2], crop_coords[3])

        crop_shapes[split].append((x_crop.stop - x_crop.start, y_crop.stop - y_crop.start))

        image = image[x_crop, y_crop]
        mask = mask[x_crop, y_crop]
        return image, mask

    with h5py.File(hdf5_file, 'r') as f:
        for cell_idx, death_frame in zip(death_frames['Cell Idx'], death_frames['Death Frame']):
            split = 'validate' if cell_idx in val_cell_ids else 'train'
            coco_json = dirs[split]['coco_json']
            images_dir = dirs[split]['images']

            num_frames, image_h, image_w = f['Images']['Phase'].shape

            # Alive frames
            for alive_frame in death_frame - alive_frames_sample:
                if alive_frame < 0:
                    pbar.update(1)
                    continue

                mask = f['Segmentations']['Phase'][alive_frame][:] == cell_idx
                if not mask.any():
                    pbar.update(1)
                    continue
                image = f['Images']['Phase'][alive_frame]

                image, mask = crop_image_and_mask(image, mask)
                
                image_file = f'{cell_idx}_{alive_frame}_alive.jpeg'
                

                seg_found = mask_funcs.add_coco_annotation(coco_json, image_file, mask, image_id, annotation_id, 1)
                if seg_found == 1:
                    plt.imsave(images_dir / image_file, image / 255, cmap='gray')
                annotation_id += 1
                image_id += 1
                pbar.update(1)

            # Dead frames
            for dead_frame in death_frame + dead_frames_sample:
                if dead_frame > SETTINGS.NUM_FRAMES:
                    pbar.update(1)
                    continue

                mask = f['Segmentations']['Phase'][dead_frame][:] == cell_idx
                if not mask.any():
                    pbar.update(1)
                    continue

                image = f['Images']['Phase'][dead_frame]

                image, mask = crop_image_and_mask(image, mask)

                image_file = f'{cell_idx}_{dead_frame}_dead.jpeg'
                seg_found = mask_funcs.add_coco_annotation(coco_json, image_file, mask, image_id, annotation_id, 2)

                if seg_found == 1:
                    plt.imsave(images_dir / image_file, image / 255, cmap='gray')

                annotation_id += 1
                image_id += 1
                pbar.update(1)

        if include_wrong_segs:
            cell_idxs = wrong_segmentations['Cell Idx']
            val_count = max(1, int(len(cell_ids) * 0.2))
            val_cell_ids = set(random.sample(list(cell_ids), val_count))

            max_attempts = 100

            for cell_idx in cell_idxs:
                split = 'validate' if cell_idx in val_cell_ids else 'train'
                coco_json = dirs[split]['coco_json']
                images_dir = dirs[split]['images']

                frames = np.nonzero(~np.isnan(f['Cells']['Phase']['Area'][:, cell_idx]))[0]
                frame = np.random.choice(frames)

                mask = f['Segmentations']['Phase'][frame] == cell_idx
                image = f['Images']['Phase'][frame]

                image, _ = crop_image_and_mask(image, mask)
                image_file = image_file = f'wrong_seg_{cell_idx}.jpeg'

                coco_json["images"].append({
                    "id": image_id,
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "file_name": image_file,
                    })
                plt.imsave(images_dir / image_file, image / 255, cmap='gray')
                image_id += 1
                pbar.update(1)

        if include_no_cell:
            for split in ['train', 'validate']:
                crops = crop_shapes[split]

                coco_json = dirs[split]['coco_json']
                images_dir = dirs[split]['images']

                num_samples = len(crops) / 2 # Get approximately same number of no cell images as alive/dead cell images
                attempts = 0
                added = 0
                max_attempts = len(crops) * 20  # avoid infinite loop   

                coco_json = dirs[split]['coco_json'] 

                while added < num_samples and attempts < max_attempts:
                    attempts += 1
                    frame = random.randint(0, num_frames - 1)           

                    h_crop, w_crop = random.choice(crops)

                    if h_crop >= image_h or w_crop >= image_w:
                        continue

                    x_0 = random.randint(0, image_h - h_crop)
                    y_0 = random.randint(0, image_w - w_crop)

                    mask_crop = f['Segmentations']['Phase'][frame][x_0:x_0+h_crop, y_0:y_0+w_crop]
                    if (mask_crop >= 0).any():
                        continue

                    image = f['Images']['Phase'][frame][x_0:x_0+h_crop, y_0:y_0+w_crop]
                    image_file = f'no_cell_{added}.jpeg'

                    plt.imsave(images_dir / image_file, image / 255, cmap='gray')
                    coco_json["images"].append({
                        "id": image_id,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "file_name": image_file,
                        })
                    
                    image_id += 1
                    added += 1
                    pbar.update(1)

    pbar.close()

    # Save COCO jsons
    for split in dirs.keys():
        with open(dirs[split]['json_file'], 'w') as f:
            json.dump(dirs[split]['coco_json'], f)

# class ClassOnlyFastRCNNOutputLayers(FastRCNNOutputLayers):
#     def __init__(self, input_shape, *, box2box_transform, num_classes, **kwargs):
#         super().__init__(
#             input_shape,
#             box2box_transform=box2box_transform,
#             num_classes=num_classes,
#             **kwargs
#         )
    # def losses(self, predictions, proposals):
    #     """
    #     Override loss to ignore background class and optionally ignore box regression loss.
    #     """
    #     scores, proposal_deltas = predictions
    #     device = scores.device

    #     # Concatenate ground truth classes from proposals
    #     gt_classes = cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0, dtype=torch.int64, device=device)

    #     # Filter out background (class == num_classes)
    #     is_foreground = gt_classes < self.num_classes
    #     scores_fg = scores[is_foreground]
    #     gt_classes_fg = gt_classes[is_foreground]

    #     if len(gt_classes_fg) == 0:
    #         # no foreground samples in this batch: return zero loss
    #         loss_cls = torch.tensor(0.0, device=device, requires_grad=True)
    #     else:
    #         loss_cls = F.cross_entropy(scores_fg, gt_classes_fg)

    #     # Option 1: ignore box regression loss completely
    #     loss_box_reg = torch.tensor(0.0, device=device, requires_grad=True)

    #     # Option 2: If you want to keep box loss, uncomment below and comment Option 1 above
    #     # proposal_boxes = [p.proposal_boxes for p in proposals]
    #     # gt_boxes = [p.gt_boxes for p in proposals]
    #     # loss_box_reg = self.box_reg_loss(proposal_deltas, proposal_boxes, gt_boxes, gt_classes)
        
    #     print(f"scores device: {scores.device}")
    #     print(f"proposal_deltas device: {proposal_deltas.device}")
    #     for p in proposals:
    #         print(f"proposal gt_classes device: {p.gt_classes.device}")

    #     return {
    #         "loss_cls": loss_cls,
    #         "loss_box_reg": loss_box_reg,
    #     }     
    
class ClassifierHeadFineTuner(train.MyTrainer):
    """Freeze everything except classification head, and remove all augemtnations in mapper (can't set augmentation sizes to image size like in MyTrainer as image crop sizes vary)."""
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, train.LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                # DatasetMapper(self.cfg, True),
                mapper = no_resize_mapper,
            )
        ))
        return hooks
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        """Freezes everything but classification head."""
        for name, param in model.named_parameters():
            if 'proposal_generator' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
            # if "roi_heads.box_predictor.cls_score" in name:
            # # if "roi_heads.box_predictor" in name:
            #     param.requires_grad = True

            # else:
            #     param.requires_grad = False
        
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.SOLVER.BASE_LR
        )
    
    # @classmethod
    # def build_model(cls, cfg):
    #     model = super().build_model(cfg)
    #     replace_box_predictor_with_custom(model, cfg)
    #     return model
    @classmethod
    def build_train_loader(cls, cfg):
        """Set mapper as no resize mapper,"""
        return build_detection_train_loader(cfg, mapper=no_resize_mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # if dataset_name is None:
        #     dataset_name = cfg.DATASETS.TEST[0]
        return build_detection_test_loader(cfg, 
                                           dataset_name,
                                        #    mapper=DatasetMapper(cfg, is_train=False),
                                           mapper=no_resize_mapper,
                                           )

# NOT USED IGNORE
class CustomWeightedROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        # Set your weights here
        self.cls_loss_weight = 1.0
        self.bbox_loss_weight = 0.1
        self.mask_loss_weight = 0.1

    def losses(self, predictions, proposals):
        """
        Override to compute weighted loss.
        """
        # This calls the original StandardROIHeads losses method
        losses = super().losses(predictions, proposals)

        # Modify weights here
        if "loss_cls" in losses:
            losses["loss_cls"] *= self.cls_loss_weight
        if "loss_box_reg" in losses:
            losses["loss_box_reg"] *= self.bbox_loss_weight
        if "loss_mask" in losses:
            losses["loss_mask"] *= self.mask_loss_weight

        return losses

def no_resize_mapper(dataset_dict):
    """
    A custom mapper that loads image and annotations *without* resizing.
    Assumes dataset_dict is in Detectron2 standard format.
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # don't modify original dict

    # Load image from file
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # augs = T.AugmentationList([
    #                         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    #                         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
    #                            ])
    augs = T.AugmentationList([])
    
    aug_input = T.AugInput(image)

    h, w = aug_input.image.shape[:2]

    transforms = augs(aug_input)
    image = torch.as_tensor(aug_input.image.transpose(2, 0, 1).copy())
    dataset_dict["image"] = image
    # image, transforms = T.apply_augmentations(augs, image)
    # Load annotations (if training)
    if "annotations" in dataset_dict:
        annos = [
            utils.transform_instance_annotations(obj, transforms, (h, w))
            for obj in dataset_dict.pop("annotations")
        ]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, (h, w))

    return dataset_dict


def fine_tune(directory=SETTINGS.MASK_RCNN_MODEL):
    """
    Fine tune the Mask R-CNNmodel on cropped cell images just before and after cell death. Freeze everything expect for classification head.
    """
    # if "CustomWeightedROIHeads" not in ROI_HEADS_REGISTRY._obj_map:
    #     ROI_HEADS_REGISTRY.register(CustomWeightedROIHeads)
    setup_logger()

    metrics_path = Path(directory / 'Model' / 'metrics.json')
    if metrics_path.exists():
        metrics_path.rename(metrics_path.with_name("original_metrics.json"))

    dataset_dir = directory / 'Fine_Tuning_Data'
    config_directory = directory / 'Model'
    register_coco_instances("my_dataset_train", {}, str(dataset_dir / 'train' / 'labels.json'), str(dataset_dir / 'train' / 'images'))
    register_coco_instances("my_dataset_val", {},str(dataset_dir / 'validate' / 'labels.json'), str(dataset_dir / 'validate' / 'images'))

    train_metadata = MetadataCatalog.get("my_dataset_train")
    
    cfg = get_cfg()
    cfg.merge_from_file(str(directory / 'Model' / 'config.yaml'))
    cfg.OUTPUT_DIR = str(directory / 'Model')
    cfg.MODEL.WEIGHTS = str(directory / 'Model' / 'model_final.pth')
    cfg.SOLVER.BASE_LR = 1e-5 # Decrease learnign rate for fine tuning (was 0.00025 for main training)
    cfg.SOLVER.IMS_PER_BATCH = 8 # Avoids issues with collating images of different sizes
    cfg.SOLVER.MAX_ITER = 500
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    # cfg.MODEL.ROI_HEADS.NAME = "CustomWeightedROIHeads"
    cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeads'
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    trainer = ClassifierHeadFineTuner(cfg)
    trainer.resume_or_load(resume=False)
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
    # mask_funcs.clean_coco_json(
    #     Path('PhagoPred/detectron_segmentation/models/27_05_mac_new_bigger_dataset/Fine_Tuning_Data/validate/labels.json'),
    #     Path('PhagoPred/detectron_segmentation/models/27_05_mac_new_bigger_dataset/Fine_Tuning_Data/validate/images'),
    #     )
    fine_tune()
    # get_finetuning_dataset()
    # plot_loss(Path("PhagoPred/detectron_segmentation/models/27_05_mac_finetune/Model"))

if __name__ == '__main__':
    main()