# detectron imports
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

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import logging
import time
import datetime
import json

from PhagoPred.utils import tools
from PhagoPred import SETTINGS

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

torch.cuda.empty_cache()

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks
    


    @classmethod
    def build_train_loader(cls, cfg):
        augmentations = [
            # Add your augmentations here, for example:
            # T.RandomRotation(angle=[-10, 10]),
            # #T.RandomBrightness(0.8, 1.2),
            # T.RandomContrast(0.8, 1.2),
            #T.RandomResize(shape_list=[(800, 800), (1200, 1200), (1600, 1600)])
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentations)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder, max_dets_per_image=1000)

    

def train(directory=SETTINGS.MASK_RCNN_MODEL):
    setup_logger()

    dataset_dir = directory / 'Training_Data'
    config_directory = directory / 'Model'
    register_coco_instances("my_dataset_train", {}, str(dataset_dir / 'train' / 'labels.json'), str(dataset_dir / 'train' / 'images'))
    register_coco_instances("my_dataset_val", {},str(dataset_dir / 'validate' / 'labels.json'), str(dataset_dir / 'validate' / 'images'))

    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    tools.remake_dir(config_directory)
    with open(str(config_directory / "train_metadata.json"), 'w') as json_file:
        json.dump(train_metadata.as_dict(), json_file)

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = str(config_directory)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")) # pretrained model to use, doesn't make much difference which you pick
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 4  
    cfg.SOLVER.BASE_LR = 0.00025 
    cfg.SOLVER.MAX_ITER = 500 
    cfg.SOLVER.STEPS = []  
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.DETECTIONS_PER_IMAGE = 500 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # The default settings apply some size augmenations to the images. I've found this decreases performance so this section makes sure all images are the same size
    with open(dataset_dir / 'train' / 'labels.json', 'r') as f:
        train_coco = json.load(f)
    min_size = min(train_coco['images'][0]['height'], train_coco['images'][0]['width'])
    max_size = max(train_coco['images'][0]['height'], train_coco['images'][0]['width'])
    cfg.INPUT.MIN_SIZE_TRAIN = (min_size,)
    cfg.INPUT.MAX_SIZE_TRAIN = max_size
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"

    cfg.INPUT.MIN_SIZE_TEST = min_size 
    cfg.INPUT.MAX_SIZE_TEST = max_size 

    cfg.TEST.AUG = cfg.TEST.AUG if "AUG" in cfg.TEST else {}
    cfg.TEST.AUG["MAX_SIZE"] = max_size 
    cfg.TEST.AUG["MIN_SIZES"] = [min_size] 

    cfg.TEST.EVAL_PERIOD = 100

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    config_yaml_path = config_directory / 'config.yaml'
    with open(str(config_yaml_path), 'w') as file:
        yaml.dump(cfg, file)

    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(config_directory / 'metrics.json')
    plt.rcParams["font.family"] = 'serif'
    plt.scatter([x['iteration'] for x in experiment_metrics if 'total_loss' in x], [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color='navy')
    plt.scatter(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], color='red')
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.savefig(config_directory / 'loss_plot.png')
    plt.clf()

def main():
    train()

if __name__ == '__main__':
    main()
