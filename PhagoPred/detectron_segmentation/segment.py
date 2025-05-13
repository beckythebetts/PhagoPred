import sys
sys.path.insert(0, 'detectron2')

import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

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

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seg_image(cfg_dir, im):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(str(cfg_dir / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(cfg_dir / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(cfg_dir / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)
    detectron_outputs = predictor(im)
    # class_masks = {class_name: torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
    #                                             device=device)
    #                for class_name in train_metadata['thing_classes']}
    mask_cell = torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                device=device)
    mask_cluster = torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                device=device)

    for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
        class_name = train_metadata['thing_classes'][pred_class]
        instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)

        # ******* ADJUST FOR NON SQUARE IMAGES*********
        # if SETTINGS.REMOVE_EDGE_CELLS:
        #     if torch.any(torch.nonzero(instance_mask)==1) or torch.any(torch.nonzero(instance_mask)==SETTINGS.IMAGE_SIZE[0]-1):
        #         continue
        if class_name == 'Amoeba':
            mask_cell = torch.where(instance_mask,
                                torch.tensor(i, dtype=torch.int16),
                                mask_cell)

        elif class_name == 'Cluster':
            mask_cluster = torch.where(instance_mask,
                                torch.tensor(i, dtype=torch.int16),
                                mask_cluster)
            
    torch.cuda.empty_cache()


    return mask_cell.cpu().numpy().astype(np.int16), mask_cluster.cpu().numpy().astype(np.int16)

def segment_tiff(tiff_file: Path, save_dir: Path, model_dir=SETTINGS.MASK_RCNN_MODEL):
    save_dir.mkdir(exist_ok=True)

    tiff_stack = tiff.imread(str(tiff_file))

    config_directory = model_dir / 'Model'
    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    for frame_idx, frame in enumerate(tiff_stack):
        sys.stdout.write(f'\rSegmenting image {frame_idx} / {len(tiff_stack)}')
        sys.stdout.flush()
        frame_8bit = (frame / frame.max() * 255).astype(np.uint8)
        frame_processed = np.stack([np.array(frame_8bit)]*3, axis=-1)
        detectron_outputs = predictor(frame_processed)

        mask = torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                        device=device)
        for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
            class_name = train_metadata["thing_classes"][pred_class]
            instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)
            if class_name == 'Amoeba':
                mask = torch.where(instance_mask, i, mask)

        mask = mask.cpu().numpy()

        mask_im = Image.fromarray(mask)

        mask_im.save(save_dir / f'{frame_idx:04}.png')


        
def main():
    print('--------------------\nSEGMENTING - ', SETTINGS.CLASSES['phase'], '\n--------------------')
    config_directory = SETTINGS.MASK_RCNN_MODEL / 'Model'
    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    with h5py.File(SETTINGS.DATASET, 'r+') as f:
        if 'Segmentations' in f:
            del f['Segmentations']
        for frame, im in f['Images']['Phase'].items():
            sys.stdout.write(f'\rSegmenting image {int(frame)+1} / {f["Images"].attrs["Number of frames"]}')
            sys.stdout.flush()
            detectron_outputs = predictor(np.stack([np.array(im)]*3, axis=-1))
            # class_masks = {class_name: torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
            #                                             device=device)
            #                for class_name in train_metadata['thing_classes']}
            mask = torch.zeros_like(detectron_outputs["instances"].pred_masks[0], dtype=torch.int16,
                                                        device=device)
            cell_idx = 2
            for i, pred_class in enumerate(detectron_outputs["instances"].pred_classes):
                class_name = train_metadata['thing_classes'][pred_class]
                instance_mask = detectron_outputs["instances"].pred_masks[i].to(device=device)
                # ******* ADJUST FOR NON SQUARE IMAGES*********
                # if SETTINGS.REMOVE_EDGE_CELLS:
                #     if torch.any(torch.nonzero(instance_mask)==1) or torch.any(torch.nonzero(instance_mask)==SETTINGS.IMAGE_SIZE[0]-1):
                #         continue
                
                if class_name == 'Cell':
                    mask = torch.where(instance_mask,
                                       torch.tensor(cell_idx, dtype=torch.int16),
                                       mask)
                    cell_idx += 1
                elif class_name == 'Cluster':
                    mask = torch.where(instance_mask,
                                       torch.tensor(1, dtype=torch.int16),
                                       mask)
            mask = mask.cpu().numpy()
            mask = f.create_dataset(f'Segmentations/Phase/{int(frame):04}', dtype='i2', data=mask)
            # for class_name, class_mask in class_masks.items():
            #     class_mask_np = class_mask.cpu().numpy()
            #     mask = f.create_dataset(f'Segmentations/Phase/{int(frame):04}', dtype='i2', data=class_mask_np)
            #     #f['Segmentations']['Phase'][i] = class_mask_np


if __name__ == '__main__':
    tiff_file = Path('temp')/ 'no_yeat_1' / 'no_yeat_1_MMStack_Pos0_3.ome.tif'
    print(f'Segmenting File {tiff_file.name}')
    segment_tiff(tiff_file, tiff_file.parent / f'{tiff_file.stem}'[:-4])
    # main()

