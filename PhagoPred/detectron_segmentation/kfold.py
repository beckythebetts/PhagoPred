import sys
import faulthandler
import warnings

from sympy import im
sys.path.insert(0, 'detectron2')

from pathlib import Path
import shutil
from subprocess import call
from detectron2.data import MetadataCatalog, DatasetCatalog
import ast
import numpy as np
from collections import OrderedDict
import json
import pycocotools
import matplotlib.pyplot as plt
import cellpose.metrics
import pandas as pd
from PIL import Image
import seaborn as sns

from PhagoPred.detectron_segmentation.train import train
from PhagoPred.detectron_segmentation.eval import evaluator, Evaluator
from PhagoPred.detectron_segmentation.segment import seg_image
from PhagoPred.detectron_segmentation import fine_tune_class
from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs



def make_coco_subset(orig_coco, ims):
    with open(orig_coco, 'r') as f:
        orig_coco = json.load(f)
        new_coco = {
        'images': [],
        'annotations': [],
        'categories': orig_coco['categories']
        }
        annotation_id = 1
        for image in orig_coco['images']:
            if image['file_name'] in [im.name for im in ims]:
                new_coco['images'].append({'id': len(new_coco['images']) + 1,
                                           'file_name': image['file_name'],
                                           'width': image['width'],
                                           'height': image['height']})
                for annotation in orig_coco['annotations']:
                    if annotation['image_id'] == image['id']:
                        new_annotation = annotation.copy()
                        new_annotation['image_id'] = len(new_coco['images'])
                        new_annotation['id'] = annotation_id
                        annotation_id += 1
                        new_coco['annotations'].append(new_annotation)
    return new_coco

class KFold:

    def __init__(self, directory, fine_tune: bool = False):
        self.directory = directory
        self.ims = sorted([im for im in (self.directory / 'images').iterdir()])
        self.coco = directory / 'labels.json'
        self.fine_tune = fine_tune
        with open(self.coco, 'r') as f:
            self.categories = [category['name'] for category in json.load(f)["categories"]]

    def split_all(self, num_val=1):
        if self.fine_tune:
            self.split_for_fine_tune()
            return 0
        
        for i in range(int(len(self.ims)/num_val)):
            training_data = self.directory / f'model_{i}' / 'Training_Data'

            tools.remake_dir(training_data / 'train' / 'images')
            tools.remake_dir(training_data / 'validate' / 'images')

            val_ims = [self.ims[((i*num_val)+x) % len(self.ims)] for x in range(num_val)]
            train_ims = [im for im in self.ims if im not in val_ims]

            val_coco = make_coco_subset(self.coco, val_ims)
            with open(training_data / 'validate' / 'labels.json', 'w') as f:
                json.dump(val_coco, f)
            train_coco = make_coco_subset(self.coco, train_ims)
            with open(training_data / 'train' / 'labels.json', 'w') as f:
                json.dump(train_coco, f)

            for im in self.ims:
                if im in val_ims:
                    shutil.copy(im, training_data / 'validate' / 'images' / im.name)

                elif im in train_ims:
                    shutil.copy(im, training_data / 'train' / 'images' / im.name)

    def split_for_fine_tune(self, num_splits: int = 3):
        """
        Split data into num_splits folds for fine tuning. Each split has a different set of validation images, and images containing the same cells are kept together.
        80:20 train val split.
        """
        all_cell_idxs = []
        for im in self.ims:
            if 'alive' in im.name or 'dead' in im.name:
                all_cell_idxs.append(im.name.split('_')[0])
        all_cell_idxs = set(all_cell_idxs)
        validation_cell_idxs = np.random.choice(list(all_cell_idxs), size=int(num_splits*0.2*len(all_cell_idxs)), replace=False)
        for i in range(num_splits):
            split_data_dir = self.directory / f'split_{i}' / 'Fine_Tuning_Data'

            tools.remake_dir(split_data_dir / 'train' / 'images')
            tools.remake_dir(split_data_dir / 'validate' / 'images')

            split_val_idxs = validation_cell_idxs[i::num_splits]
            for im in self.ims:
                if 'alive' in im.name or 'dead' in im.name:
                    cell_idx = im.name.split('_')[0]
                    if cell_idx in split_val_idxs:
                        shutil.copy(im, split_data_dir / 'validate' / 'images' / im.name)
                    else:
                        shutil.copy(im, split_data_dir / 'train' / 'images' / im.name)
                else:
                    val = np.random.rand() < 0.2
                    if val:
                        shutil.copy(im, split_data_dir / 'validate' / 'images' / im.name)   
                    else:
                        shutil.copy(im, split_data_dir / 'train' / 'images' / im.name)

            # Collect names s all training and validation images
            train_ims = [im for im in (split_data_dir / 'train' / 'images').iterdir()]
            val_ims = [im for im in (split_data_dir / 'validate' / 'images').iterdir()]

            # Make COCO subsets and save
            train_coco = make_coco_subset(self.coco, train_ims)
            with open(split_data_dir / 'train' / 'labels.json', 'w') as f:
                json.dump(train_coco, f)
            val_coco = make_coco_subset(self.coco, val_ims)
            with open(split_data_dir / 'validate' / 'labels.json', 'w') as f:
                json.dump(val_coco, f)
            
            # Copy original model into each split directory
            shutil.copytree(self.directory.parent / 'Model', split_data_dir.parent / 'Model')



    
        # for names in [('00', '01'), ('01', '10'), ('10', '11'), ('11', '10')]:
        #     self.make_split(names)

    def train(self):
        if self.fine_tune:
            for file in self.directory.glob('split_*'):
                fine_tune_class.fine_tune(directory=file)
                unregister_coco_instances('my_dataset_train')
                unregister_coco_instances('my_dataset_val')
        else:
            for file in self.directory.glob('*model_*'):
                # if 'model_3' not in file.name:
                train(directory=file)
                unregister_coco_instances('my_dataset_train')
                unregister_coco_instances('my_dataset_val')
                evaluator(directory=file)
                unregister_coco_instances('my_dataset_train')
                unregister_coco_instances('my_dataset_val')

    def eval(self):
        for file in self.directory.glob('*split_*'):
            for im_name in (file / 'Fine_Tuning_Data' / 'validate' / 'images').iterdir():
                im = plt.imread(im_name)

                if im.ndim == 2:
                    frame_processed = np.stack([im]*3, axis=-1)
                else:
                    frame_processed = im

                # frame_processed = np.stack([np.array(im)]*3, axis=-1)
                print(frame_processed.shape)
                pred_masks = seg_image(cfg_dir = file / 'Model', im = frame_processed)
                true_masks = mask_funcs.coco_to_masks(coco_file = file / 'Fine_Tuning_Data' / 'validate' / 'labels.json', im_name = im_name) 
                
                if pred_masks is not None:
                    combi_pred = mask_funcs.combine_masks(list(pred_masks.values()))
                else: 
                    combi_pred = np.zeros(im.shape[:2], dtype=np.uint8)
                
                if true_masks is not None:
                    combi_true = mask_funcs.combine_masks(list(true_masks.values()))
                else:
                    combi_true = np.zeros(im.shape[:2], dtype=np.uint8)

                view_all = tools.show_segmentation(im, combi_pred, combi_true)
                plt.imsave(file / f'{im_name.stem}_view.png', view_all/255)

                pred_mask_im = Image.fromarray(combi_pred.astype(np.int32), mode='I')
                pred_mask_im.save(file / f'{im_name.stem}_pred_mask.png')
                
                results = self.prec_recall_curve(combi_true, combi_pred)
                results.to_csv(str(file / f'{im_name.stem}_all_results.txt'), sep='\t')

                for category in pred_masks.keys():
                    view = tools.show_segmentation(im, pred_masks[category], true_masks[category])
                    plt.imsave(file / f'{im_name.stem}_{category}_view.png', view)

                    pred_mask_im = Image.fromarray(pred_masks[category].astype(np.int32), mode='I')
                    pred_mask_im.save(file / f'{im_name.stem}_{category}_pred_mask.png')

                    results = self.prec_recall_curve(true_masks[category], pred_masks[category])
                    results.to_csv(str(file / f'{im_name.stem}_{category}_results.txt'), sep='\t')
    
    def fine_tune_eval_clusters(self, plot_title: str):
        """Plot precision recall curves of clusters."""
        results = []
        for file in self.directory.glob('split_*'):
            metric_evaluator = Evaluator(file / 'Fine_Tuning_Data', file / 'Model', eval_mode='metrics')
            metric_evaluator.eval()
            for results_file in (file / 'Evaluation').glob("*_results.txt"):
                results.append(pd.read_csv(results_file, sep = '\t', index_col = 0))
        results = pd.concat(results, axis=0)
        means, stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
        percentiles = results.groupby(level=0).quantile([0.05, 0.5, 0.95]).unstack(level=1)
        metrics = means.columns.values
        thresholds = means.index.values
        
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axs, metrics):
            p5 = percentiles[(metric, 0.05)]
            median = percentiles[(metric, 0.5)]
            p95 = percentiles[(metric, 0.95)]
            
            # ax.plot(thresholds, means[metric], color=colour, label=label)
            # ax.fill_between(thresholds, means[metric]-stds[metric], means[metric]+stds[metric], color=colour, alpha=0.5, edgecolor='none')
            
            ax.plot(thresholds, median)
            ax.fill_between(thresholds, p5, p95, alpha=0.5, edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
        plt.title(plot_title)
        # plt.legend()
        plt.savefig(self.directory / 'cluster_results.png')
        
    def fine_tune_eval(self, cm_title: str):
        cms = []
        accs = []
        for file in self.directory.glob('split_*'):
            evaluator = Evaluator(file / 'Fine_Tuning_Data', file / 'Model', eval_mode="confusion")
            cm, acc = evaluator.eval()
            cms.append(cm)
            accs.append(acc)

        cms = np.array(cms)
        mean_cm = np.mean(cms, axis=0)
        std_cm = np.std(cms, axis=0)
        
        median_cm = np.median(cms, axis=0)
        p5_cm = np.percentile(cms, 5, axis=0)
        p95_cm = np.percentile(cms, 95, axis=0)

        # Create annotation strings like "0.85±0.02"
        annotations = np.empty_like(mean_cm, dtype=object)
        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                # annotations[i, j] = f"{mean_cm[i, j]:.2f}±{std_cm[i, j]:.2f}"
                annotations[i, j] = f"{median_cm[i, j]:.2f} ({p5_cm[i, j]:.2f}–{p95_cm[i, j]:.2f})"

        labels = ['Macrophage', 'Dead Macrophage', 'No Cell']
        # Plot
        plt.figure(figsize=(6, 6))
        sns.heatmap(median_cm, annot=annotations, fmt='', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=False)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(cm_title)
        no_cell_idx = 2
        plt.axhline(no_cell_idx, color='gray', linestyle='--', linewidth=2)
        plt.axvline(no_cell_idx, color='gray', linestyle='--', linewidth=2)
        # plt.axhline(no_cell_idx + 1, color='gray', linestyle='--', linewidth=2)
        # plt.axvline(no_cell_idx + 1, color='gray', linestyle='--', linewidth=2)
        plt.tight_layout()
        plt.savefig(self.directory / "confusion_matrix_mean_std.png")

        print(f'Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}')
    
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
        precisions = TPs / (TPs+FPs)
        recalls = TPs / (TPs+FNs)
        F1s = TPs / (TPs + 0.5*(FPs+FNs))
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
        plt.savefig(self.directory / 'results.png')

    def plot_results(self):
        results = []
        ious = []
        num_training_cells = []
        for dir in self.directory.glob('*model*'):
            for file in (dir / 'Training_Data' / 'validate').glob('*_results.txt'):
                results.append(pd.read_csv(file, sep = '\t', index_col = 0))
            for file in (dir / 'Training_Data' / 'validate').glob('*_iou.txt'):
                with open(file, 'r') as f:
                    ious.append(float(f.read()))
            with open(dir / 'number_training_cells.txt') as f:
                num_training_cells.append(int(f.read()))
        # results = [pd.read_csv(file, sep='\t', index_col=0) for file in self.directory.iterdir() if file.suffix=='.txt']
        results = pd.concat(results, axis=0)
        means, stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
        metrics = means.columns.values
        thresholds = means.index.values

        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for ax, metric in zip(axs, metrics):
            # ax.plot(thresholds, cellpose_means[metric], color='red', label='Cellpose')
            # ax.fill_between(thresholds, cellpose_means[metric]-cellpose_stds[metric], cellpose_means[metric]+cellpose_stds[metric], color='red', alpha=0.5, edgecolor='none')
            ax.plot(thresholds, means[metric], color='navy', label='Mask R-CNN')
            ax.fill_between(thresholds, means[metric]-stds[metric], means[metric]+stds[metric], color='navy', alpha=0.5, edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric)
            ax.grid(True)
        plt.legend()
        # plt.show()
        plt.savefig(self.directory / 'results.png')
        print(ious)
        with open(self.directory / 'info.txt', 'w') as file:
            file.write(f'IOUS = {np.mean(ious)} +- {np.std(ious)}\nNumber of training cells = {np.mean(num_training_cells)} +- {np.std(num_training_cells)}')

    def features_scatter_plot(self, feature_func=mask_funcs.get_areas, feature_name='area/pixels'):
        all_areas_true = []
        all_ious_true = []

        all_areas_pred = []
        all_ious_pred = []
        for file in self.directory.glob('*model*'):               
            for im_name in (file / 'Training_Data' / 'validate' / 'images').iterdir():
                print(im_name)
                with Image.open(file / f'{im_name.stem}_pred_mask.png') as im:
                    pred_mask = np.array(im)
                true_masks = mask_funcs.coco_to_masks(coco_file = file / 'Training_Data' / 'validate' / 'labels.json', im_name = im_name) 

                true_mask = mask_funcs.combine_masks(list(true_masks.values()))
                true_areas = feature_func(true_mask)
                pred_areas = feature_func(pred_mask)

                all_areas_true += list(true_areas)
                all_areas_pred += list(pred_areas)

                for i, area in enumerate(true_areas):
                    true_cell = np.where(true_mask==i, True, False)
                    pred_idxs = np.unique(pred_mask[true_cell])
                    iou = np.max([mask_funcs.cal_iou(true_cell, np.where(pred_mask==pred_idx, True, False)) for pred_idx in pred_idxs])
                    all_ious_true += [iou]

                for i, area in enumerate(pred_areas):
                    pred_cell = np.where(pred_mask==i, True, False)
                    true_idxs = np.unique(true_mask[pred_cell])
                    iou = np.max([mask_funcs.cal_iou(pred_cell, np.where(true_mask==true_idx, True, False)) for true_idx in true_idxs])
                    all_ious_pred += [iou]

        plt.rcParams["font.family"] = 'serif'
        plt.scatter(all_areas_true, all_ious_true, marker='.', color='r', edgecolors='none', alpha=0.5, label=f'True Cell {feature_name}')
        plt.scatter(all_areas_pred, all_ious_pred, marker='.', color='b', edgecolors='none', alpha=0.5, label=f'Predicted Cell {feature_name}')
        plt.ylabel('Intersection over Union')
        plt.xlabel(feature_name)
        plt.legend()
        plt.grid()
        plt.savefig(self.directory / f'per_ar_scatter.png')

    def eval_feature(self, feature_func=mask_funcs.get_areas, feature_name='area/pixels', n_groups=4):
        # calculte area percentiles based on true masks
        all_areas = np.array([])
        num_ims = 0
        for file in self.directory.glob('*model*'):
            for im_name in (file / 'Training_Data' / 'validate' / 'images').iterdir():
                true_masks = mask_funcs.coco_to_masks(coco_file = file / 'Training_Data' / 'validate' / 'labels.json', im_name = im_name) 
                true_mask = mask_funcs.combine_masks(list(true_masks.values()))
                all_areas = np.append(all_areas, feature_func(true_mask))
                num_ims += 1

        area_thresholds = np.percentile(all_areas, np.linspace(0, 100, n_groups+1))
        area_thresholds[0] = 0
        area_thresholds[-1] = np.inf

        # split masks
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        metrics = ['Precision', 'Recall', 'F1']
        area_threshold_names = [f'{low:.4f} - {high:.4f}' for low, high in zip(area_thresholds[:-1], area_thresholds[1:])]

        print(area_threshold_names)

        results = pd.DataFrame(columns=pd.MultiIndex.from_product([metrics, area_threshold_names, iou_thresholds], 
                                                                  names = ['Metric', 'Area Threshold', 'IOU Threshold']),
                                                                  index=np.arange(num_ims))
        
        results = results.sort_index(level=['Metric', 'Area Threshold', 'IOU Threshold'])
    
        row = 0
        for file in self.directory.glob('*model*'):
            for im_name in (file / 'Training_Data' / 'validate' / 'images').iterdir():

                # pred_mask = plt.imread(file / 'Training_Data' / 'validate' / f'{im_name.stem}_pred_mask.png')
                with Image.open(file / f'{im_name.stem}_pred_mask.png') as im:
                    pred_mask = np.array(im)
                true_masks = mask_funcs.coco_to_masks(coco_file = file / 'Training_Data' / 'validate' / 'labels.json', im_name = im_name) 
                true_mask = mask_funcs.combine_masks(list(true_masks.values()))
                true_areas, pred_areas = feature_func(true_mask), feature_func(pred_mask)

                for col, (lower_threshold, upper_threshold) in enumerate(zip(area_thresholds[:-1], area_thresholds[1:])):
                    
                    true_idxs = np.where((true_areas >= lower_threshold) & (true_areas < upper_threshold))[0]
                    pred_idxs = np.where((pred_areas >= lower_threshold) & (pred_areas < upper_threshold))[0]

                    split_true_mask = np.where(np.isin(true_mask, true_idxs), true_mask, 0)
                    split_true_mask = mask_funcs.squash_idxs(split_true_mask)

                    split_pred_mask = np.where(np.isin(pred_mask, pred_idxs), pred_mask, 0)    
                    split_pred_mask = mask_funcs.squash_idxs(split_pred_mask)        

                    unt, cto = np.unique(split_true_mask[split_true_mask>0], return_counts=True)
                    unp, ctp = np.unique(split_pred_mask[split_pred_mask>0], return_counts=True)

                    # print(np.min(cto), np.max(cto))
                    # print(np.min(ctp), np.max(ctp))

                    # print(lower_threshold, upper_threshold)

                    # plt.matshow(np.concatenate((np.concatenate((true_mask>0, pred_mask>0), axis=1), np.concatenate((split_true_mask>0, split_pred_mask>0), axis=1)), axis=0))
                    # plt.show()

                    # calculate precision using subset of predicted cells and all true cells, calaulcte recall using subset of true cells, all predicted cells
                    # APs, TPs, FPs, FNs = cellpose.metrics.average_precision(split_true_mask.astype(int), split_pred_mask.astype(int), threshold=iou_thresholds)
                    _, TPs, FPs, FNs = cellpose.metrics.average_precision(true_mask.astype(int), split_pred_mask.astype(int), threshold=iou_thresholds)
                    precisions = TPs / (TPs+FPs)
                    # assert all(TPs+FPs) == len(np.unique(split_pred_mask))-1, f"{TPs}, {FPs}, {len(np.unique(split_pred_mask))}, {len(np.unique(true_mask))}"
                    
                    _, TPs, FPs, FNs = cellpose.metrics.average_precision(split_true_mask.astype(int), pred_mask.astype(int), threshold=iou_thresholds)          
                    recalls = TPs / (TPs+FNs)

                    F1s = 2/((1/recalls)+(1/precisions))


                    # APs, TPs, FPs, FNs = cellpose.metrics.average_precision(split_true_mask.astype(int), split_pred_mask.astype(int), threshold=iou_thresholds)
                    # precisions = TPs / (TPs+FPs)
                    # recalls = TPs / (TPs+FNs)
                    # F1s = TPs / (TPs + 0.5*(FPs+FNs))

                    with warnings.catch_warnings():

                        warnings.filterwarnings('ignore', category=UserWarning, message=".*lexsort depth*")

                        results.loc[row, ('Precision', area_threshold_names[col])] = precisions
                        # results.xs(('Precision', lower_threshold), axis=1, level=['Metric', 'Area Threshold']).iloc[row] = precisions
                        results.loc[row, ('Recall', area_threshold_names[col])] = recalls
                        # results.xs(('Recall', lower_threshold), axis=1, level=['Metric', 'Area Threshold']).iloc[row] = recalls
                        results.loc[row, ('F1', area_threshold_names[col])] = F1s
                        # results.xs(('F1', lower_threshold), axis=1, level=['Metric', 'Area Threshold']).iloc[row] = F1s
                row += 1

        results_means = results.mean()
        results_stds = results.std()
        # plot
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        cmap = plt.get_cmap('viridis')
        for ax, metric in zip(axs, metrics):
            for i, area_threshold_name in enumerate(area_threshold_names):
                means = results_means[(metric, area_threshold_name)].to_numpy().astype(np.float32)
                stds = results_stds[(metric, area_threshold_name)].to_numpy().astype(np.float32)

                ax.plot(iou_thresholds, means, color=cmap((i+1)/n_groups), label=area_threshold_name)
                ax.fill_between(iou_thresholds, means-stds, means+stds, color=cmap((i+1)/n_groups), alpha=0.5, edgecolor='none')
                ax.set_xlabel('IOU Threshold')
                ax.set_ylabel(metric)
                ax.grid(True)
        plt.legend(title=feature_name)
        plt.savefig(self.directory / f'{feature_name}_results.png')

    def plot_category(self, category: str, axs: plt.Axes, colour, label: 'str' = None) -> None:
        """
        Plot precision-recall curves for given category on axs.
        """
        if label is None:
            label = category
        results = []
        for dir in self.directory.glob('*model*'):
            for file in (dir).glob(f"*{category}_results.txt"):
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
                
    def getAP(self, file):
        try:
            with open(file / 'Model' / 'eval.txt', 'r') as f:
                AP_string = f.read()

            AP_string = AP_string.replace('nan', 'np.nan').strip()
            AP_dict = eval(AP_string)
            return AP_dict[0]['segm']['AP']
        except ValueError as e:
            print("Error:", e)


    def plot_av_loss(self, output_path: Path) -> None:
        list_of_paths = [file / 'Model' / 'metrics.json' for file in self.directory.glob('*model*') if file.is_dir()]
        def load_json_arr(file_path):
            with open(file_path, 'r') as f:
                return [json.loads(line) for line in f]

        training_losses = {}
        validation_losses = {}

        for path in list_of_paths:

            experiment_metrics = load_json_arr(path)

            for x in experiment_metrics:
                iter = x.get('iteration')
                if iter is None:
                    continue

                if 'total_loss' in x:
                    training_losses.setdefault(iter, []).append(x['total_loss'])
                if 'validation_loss' in x:
                    validation_losses.setdefault(iter, []).append(x['validation_loss'])

        def compute_stats(loss_dict):
            iterations = sorted(loss_dict.keys())
            means = [np.mean(loss_dict[it]) for it in iterations]
            stds = [np.std(loss_dict[it]) for it in iterations]
            return iterations, means, stds

        train_iters, train_mean, train_std = compute_stats(training_losses)
        val_iters, val_mean, val_std = compute_stats(validation_losses)

        plt.rcParams["font.family"] = 'serif'
        plt.plot(train_iters, train_mean, color='navy', label='Train Loss (mean)')
        plt.fill_between(train_iters, 
                        np.array(train_mean) - np.array(train_std),
                        np.array(train_mean) + np.array(train_std),
                        color='navy', alpha=0.2)
        plt.plot(val_iters, val_mean, color='orange', label='Validation Loss (mean)')
        plt.fill_between(val_iters,
                        np.array(val_mean) - np.array(val_std),
                        np.array(val_mean) + np.array(val_std),
                        color='orange', alpha=0.2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path)

def unregister_coco_instances(name):
    if name in DatasetCatalog.list():
        DatasetCatalog.pop(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.pop(name)

def merge_jsons(directory):
    all_jsons = [f for f in directory.glob('*.json')]
    json_0 = all_jsons[0]
    for i in range(1, len(all_jsons)):
        call(['python', '-m', 'COCO_merger.merge', '--src', json_0, all_jsons[i], '--out', directory / 'labels.json'])
        json_0 = directory / 'labels.json'

# def plot_comparison(kfold_dict, save_as):
#     colours = ['red', 'navy']
#     plt.rcParams["font.family"] = 'serif'
#     fig, axs = plt.subplots(1, 3, figsize=(12, 4))
#     for (name, kfold), colour in zip(kfold_dict.items(), colours):
#         kfold.plot_multiple(name, axs, colour)
#     plt.legend()
#     plt.savefig(save_as)

def plot_comparison(kfols_dict, save_as, category):
    colours = ['red', 'navy']
    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for (name, kfold), colour in zip(kfols_dict.items(), colours):
        kfold.plot_category(category, axs, colour, label=name)
    plt.legend()
    plt.suptitle(f'Comparison of {category} category')
    plt.savefig(save_as)

def split_masks_by_area(true_masks, pred_masks, n_groups=4):
    '''
    Split both sets of masks into n_groups based on size. Thresholds are determined by evenly spaced percentiles of true_mask
    '''
    # calculate are area thresholds
    all_areas = np.array([])
    for mask in true_masks:
        areas = mask_funcs.get_areas(mask)
        all_areas = np.append(all_areas, areas)
    
    area_thresholds = np.percentile(all_areas, np.linspace(0, 100, n_groups+1))
    # split masks
    split_true_masks, split_pred_masks = [], []
    for true_mask, pred_mask in zip(true_masks, pred_masks):
        true_areas, pred_areas = mask_funcs.get_areas(true_mask), mask_funcs.get_areas(pred_mask)
        for lower_threshold, upper_threshold in zip(area_thresholds[:-1], area_thresholds[1:]):
            true_idxs = np.where((true_areas >= lower_threshold) & (true_areas < upper_threshold)) 
            pred_ixs = np.where((true_areas >= lower_threshold) & (true_areas < upper_threshold))                                     

def compare_prec_recall_curves(directories: list, save_as: Path, labels: list) -> None:
    cmap = plt.cm.get_cmap('Set1')
    plt.rcParams["font.family"] = 'serif'
    colours = [cmap(i) for i in range(len(labels))]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, directory in enumerate(directories):
        results = []
        for file in directory.glob('split_*'):
            # metric_evaluator = Evaluator(file / 'Fine_Tuning_Data', file / 'Model', eval_mode='metrics')
            # metric_evaluator.eval()
            for results_file in (file / 'Evaluation').glob("*_results.txt"):
                results.append(pd.read_csv(results_file, sep = '\t', index_col = 0))
        results = pd.concat(results, axis=0)
        means, stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
        percentiles = results.groupby(level=0).quantile([0.05, 0.5, 0.95]).unstack(level=1)
        metrics = means.columns.values
        thresholds = means.index.values
    
        for ax, metric in zip(axs, metrics):
            p5 = percentiles[(metric, 0.05)]
            median = percentiles[(metric, 0.5)]
            p95 = percentiles[(metric, 0.95)]
            
            # ax.plot(thresholds, means[metric], color=colour, label=label)
            # ax.fill_between(thresholds, means[metric]-stds[metric], means[metric]+stds[metric], color=colour, alpha=0.5, edgecolor='none')
            
            ax.plot(thresholds, median, color=colours[i], label=labels[i])
            ax.fill_between(thresholds, p5, p95, color=colours[i], alpha=0.5, edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
    plt.legend()
    plt.tight_layout()#
    plt.savefig(save_as)

def main():
    faulthandler.enable()
    # merge_jsons(Path('PhagoPred')/ 'detectron_segmentation' / 'models' / 'toumai_01_05' / 'labels')
    # compare_prec_recall_curves([
    #     Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'kfold_no_fine_tune_16_10',
    #     Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'kfold_fine_tune_16_10'
    # ],
    #                            Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'comparison_plot.png',
    #                            ['No Fine Tune', 'Fine Tune'])
    my_kfold = KFold(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'kfold_no_fine_tune_16_10', fine_tune=True)
    # my_kfold.split_all()
    # my_kfold.train()
    # # my_kfold.eval()

    my_kfold.fine_tune_eval('Not Fine Tuned')
    # my_kfold.fine_tune_eval_clusters('Fine Tuned')
    # my_kfold.train()
    # my_kfold.eval()
    # my_kfold.plot()

    # my_kfold.plot_results()
    # my_kfold.eval_feature(feature_func=mask_funcs.get_perimeters_over_areas, feature_name='perim_over_area', n_groups=3)
    # my_kfold.features_scatter_plot(feature_func=mask_funcs.get_perimeters_over_areas, feature_name='perimeter/area')

    # plot_comparison(
    #     {'$L_{mask}$ x 0.3': KFold(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'kfold_low_mask_loss'),
    #      '$L_{mask}$ x 1.0': KFold(Path('PhagoPred') / 'detectron_segmentation' / 'models' / '27_05_mac' / 'kfold_custom_loss')},
    #      save_as=Path('temp') / 'comparison.png',
    #      category='all')

    # my_kfold.plot_av_loss(output_path=Path('temp') / 'av_loss_plot.png')

if __name__=='__main__':
    main()