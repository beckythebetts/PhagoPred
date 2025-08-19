import sys
sys.path.insert(0, 'detectron2')

import torch, detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from cellpose import io, metrics
from PIL import Image

import numpy as np
import os, json, cv2, random, shutil
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import cellpose
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.measure import label

from PhagoPred import SETTINGS
from PhagoPred.detectron_segmentation import segment
from PhagoPred.utils import mask_funcs, tools

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluator(directory=SETTINGS.MASK_RCNN_MODEL):
    setup_logger()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    dataset_dir =  directory / 'Training_Data'
    config_directory = directory / 'Model'

    register_coco_instances("my_dataset_val", {}, str(dataset_dir / 'validate' / 'labels.json'),
                            str(dataset_dir / 'validate' / 'images'))

    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")

    with open(str(config_directory / 'train_metadata.json')) as json_file:
      train_metadata = json.load(json_file)
    cfg = get_cfg()
    cfg.merge_from_file(str(config_directory / 'config.yaml'))
    cfg.MODEL.WEIGHTS = str(config_directory / 'model_final.pth') # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", output_dir="./output", max_dets_per_image=1000)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")
    output = inference_on_dataset(predictor.model, val_loader, evaluator), 2
    print(output)
    with open(str(config_directory / 'eval.txt'), 'w') as f:
        f.write(str(output))


class Evaluator:
  def __init__(self, dataset_dir: Path = (SETTINGS.MASK_RCNN_MODEL / 'Fine_Tuning_Data'), model_dir : Path= (SETTINGS.MASK_RCNN_MODEL / 'Model'), eval_mode: str = "metrics") -> None:
    self.dataset_dir = dataset_dir
    self.model_dir = model_dir
    self.eval_dir = self.model_dir.parent / 'Evaluation'
    self.eval_mode = eval_mode  # "metrics" or "confusion"

    with open(self.dataset_dir / 'validate' / 'labels.json') as f:
        self.categories = [category['name'] for category in json.load(f)["categories"]]

    if self.eval_mode == "confusion":
        self.categories.append("No Cell")
    

  def eval(self) -> None:
    if self.eval_mode == "metrics":
        tools.remake_dir(self.eval_dir)

    y_true = []
    y_pred = []

    # Load cfg, train_metadata and predictor to avoid repated loading for each image
    train_metadata, cfg = segment.get_model(cfg_dir=self.model_dir)
    cfg, predictor = segment.get_predictor(cfg)

    for im_name in tqdm(list((self.dataset_dir / 'validate' / 'images').iterdir()), desc="Evaluating "):
        im = plt.imread(im_name)

        if im.ndim == 2:
            frame_processed = np.stack([im]*3, axis=-1)
        else:
            frame_processed = im

        pred_masks = segment.seg_image(cfg_dir=self.model_dir, im=frame_processed, train_metadata=train_metadata, cfg=cfg, predictor=predictor)
        
        if pred_masks is None:
        #   print(f'No Segmentations found for image {im_name}')
            pred_masks = {category: np.zeros_like(frame_processed[:, :, 0]) for category in self.categories}
        true_masks = mask_funcs.coco_to_masks(coco_file=self.dataset_dir / 'validate' / 'labels.json', im_name=im_name)

        if self.eval_mode == "metrics":
            self._eval_metrics(im_name, im, pred_masks, true_masks)
            self.plot()

        elif self.eval_mode == "confusion":
            # true_label = self._mask_to_class(true_masks)
            # pred_label = self._mask_to_class(pred_masks)
            true_label = self._largest_mask_to_class(true_masks)
            pred_label = self._largest_mask_to_class(pred_masks)
            y_true.append(true_label)
            y_pred.append(pred_label)

    if self.eval_mode == 'metrics':
            self.plot()
    elif self.eval_mode == 'confusion':
        self.plot_confusion_matrix(y_true, y_pred)
       

  def _eval_metrics(self, im_name, im, pred_masks, true_masks):
    combi_pred = mask_funcs.combine_masks(list(pred_masks.values()))
    combi_true = mask_funcs.combine_masks(list(true_masks.values()))

    view_all = tools.show_segmentation(im, combi_pred, combi_true)
    plt.imsave(self.eval_dir / f'{im_name.stem}_view.png', view_all / 255)

    pred_mask_im = Image.fromarray(combi_pred.astype(np.int32), mode='I')
    pred_mask_im.save(self.eval_dir / f'{im_name.stem}_pred_mask.png')

    results = self.prec_recall_curve(combi_true, combi_pred)
    results.to_csv(self.eval_dir / f'{im_name.stem}_all_results.txt', sep='\t')

    for category in pred_masks.keys():
        view = tools.show_segmentation(im, pred_masks[category], true_masks[category])
        plt.imsave(self.eval_dir / f'{im_name.stem}_{category}_view.png', view)

        pred_mask_im = Image.fromarray(pred_masks[category].astype(np.int32), mode='I')
        pred_mask_im.save(self.eval_dir / f'{im_name.stem}_{category}_pred_mask.png')

        results = self.prec_recall_curve(true_masks[category], pred_masks[category])
        results.to_csv(self.eval_dir / f'{im_name.stem}_{category}_results.txt', sep='\t')

  def _mask_to_class(self, mask_dict: dict) -> int:
    """
    Convert one-hot mask dict to a class index based on any positive prediction.
    Assumes mutually exclusive masks (non-overlapping).
    """
    for idx, category in enumerate(self.categories[:-1]):
        if np.any(mask_dict[category]):
            return idx
    return len(self.categories) - 1  # No object
  
  def _largest_mask_to_class(self, mask_dict: dict) -> int:
        """
        Given a dict of category -> binary mask, finds the largest connected predicted segment
        across all categories and returns its class index.
        If no predicted segment, returns "No Cell" class index (last index).
        """
        largest_area = 0
        largest_class_idx = len(self.categories) - 1  # Default to "No Cell"

        for idx, category in enumerate(self.categories[:-1]):  # exclude "No Cell"
            binary_mask = mask_dict[category]
            if np.any(binary_mask):
                # Label connected components in this category mask
                labeled_mask = label(binary_mask)
                # Get areas of all connected components
                for region_label in range(1, labeled_mask.max() + 1):
                    region_area = np.sum(labeled_mask == region_label)
                    if region_area > largest_area:
                        largest_area = region_area
                        largest_class_idx = idx

        return largest_class_idx

  def plot_confusion_matrix(self, y_true: list[int], y_pred: list[int]):
    plt.rcParams["font.family"] = 'serif'
    labels = list(range(len(self.categories)))
    labels_display = self.categories

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    accuracy = cm.trace() / cm.sum()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_display)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_ylabel("True label")
    
    no_cell_idx = len(self.categories) - 1
    ax.axhline(no_cell_idx - 0.5, color='gray', linestyle='--', linewidth=1)
    ax.axvline(no_cell_idx - 0.5, color='gray', linestyle='--', linewidth=1)
    ax.axhline(no_cell_idx + 0.5, color='gray', linestyle='--', linewidth=1)
    ax.axvline(no_cell_idx + 0.5, color='gray', linestyle='--', linewidth=1)

    # Optional: add text note
    # ax.text(no_cell_idx, -1.5, '← No Cell Predictions', ha='center', va='center', fontsize=8, color='gray')
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title("Confusion Matrix - No Fine Tuning")
    plt.tight_layout()
    plt.savefig(self.model_dir.parent / 'confusion_matrix.png')
    plt.close()

    print(f'Accuracy: {accuracy:4f}')

      
  def prec_recall_curve(self, true_mask: np.ndarray, 
                        pred_mask: np.ndarray, 
                        thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05)
                        ) -> pd.DataFrame:
      """
      Computes pandas datframe representing preciosn, recall and f1 curves over iou threshold sspecifed
      Parameters:
          true_mask:
          pred_mask:
      Returns:
          thresholds
      """
      APs, TPs, FPs, FNs = cellpose.metrics.average_precision(
          true_mask.astype(int), 
          pred_mask.astype(int), 
          threshold=thresholds
          )
      # precisions = TPs / (TPs+FPs)
      # recalls = TPs / (TPs+FNs)
      # F1s = TPs / (TPs + 0.5*(FPs+FNs))
      precision_denom = TPs + FPs
      recall_denom = TPs + FNs
      f1_denom = TPs + 0.5 * (FPs + FNs)

      # If there's nothing in ground truth or prediction, assign 1.0 (perfect)
      no_truth_no_pred = (TPs == 0) & (FPs == 0) & (FNs == 0)

      precisions = np.where(no_truth_no_pred, 1.0, np.divide(TPs, precision_denom, out=np.zeros_like(TPs, dtype=float), where=precision_denom != 0))
      recalls = np.where(no_truth_no_pred, 1.0, np.divide(TPs, recall_denom, out=np.zeros_like(TPs, dtype=float), where=recall_denom != 0))
      F1s = np.where(no_truth_no_pred, 1.0, np.divide(TPs, f1_denom, out=np.zeros_like(TPs, dtype=float), where=f1_denom != 0))

      df = pd.DataFrame({'Precision': precisions,
                      'Recall': recalls,
                      'F1': F1s},
                      index=thresholds)
      return df

  def plot(self) -> None:
      """
      Plot precision-recall and f1 curves of each category + all categories
      """
      plt.rcParams["font.family"] = 'serif'
      fig, axs = plt.subplots(1, 3, figsize=(12, 4))
      cmap = plt.cm.get_cmap('gist_rainbow')
      colours = [cmap(i / (len(self.categories)+1)) for i in range(len(self.categories)+1)]
      for category, colour in zip(self.categories+['all'], colours):
          self.plot_category(category, axs, colour)
      plt.legend()
      plt.savefig(self.model_dir.parent / 'results.png')

  def plot_category(self, category: str, axs: plt.Axes, colour, label: 'str' = None) -> None:
    """
    Plot precision-recall curves for given category on axs.
    """
    if label is None:
        label = category
    results = []

    for file in (self.eval_dir).glob(f"*{category}_results.txt"):
        results.append(pd.read_csv(file, sep = '\t', index_col = 0))
    results = pd.concat(results, axis=0)
    means, stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
    metrics = means.columns.values
    thresholds = means.index.values
    
    for ax, metric in zip(axs, metrics):
        ax.plot(thresholds, means[metric], color=colour, label=label)
        ax.fill_between(thresholds, means[metric]-stds[metric], means[metric]+stds[metric], color=colour, alpha=0.5, edgecolor='none')
        ax.set_xlabel('IOU Threshold')
        ax.set_ylabel(metric)
        ax.grid(True)

def main():
    # evaluator = Evaluator(dataset_dir=Path("/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/27_05_mac_finetune/Fine_Tuning_Data"), 
    #                       model_dir=Path("PhagoPred/detectron_segmentation/models/27_05_mac/Model"),
    #                       eval_mode="confusion")
    evaluator = Evaluator(
        # dataset_dir=Path("/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/27_05_mac_new/Fine_Tuning_Data"), 
        # model_dir=Path("PhagoPred/detectron_segmentation/models/27_05_mac_new/Model"),
        eval_mode="confusion",
        )
    evaluator.eval()

if __name__ == '__main__':
    main()
