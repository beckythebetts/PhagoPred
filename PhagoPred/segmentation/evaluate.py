from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

from PhagoPred.utils import mask_funcs, tools


class Evaluator:

    def __init__(self,
                 dataset_dir: Path,
                 model_dir: Path,
                 model_type: str,
                 categories: list = None):
        """
        Args:
            dataset_dir: contains train/ and validate/ subdirs, each with images/ and labels.json
            model_dir:   where model weights live (fold_dir/Model)
            model_type:  'detectron' or 'cellpose'
            categories:  list of category names; read from labels.json if not provided
        """
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.model_type = model_type
        self.eval_dir = model_dir.parent / 'Evaluation'

        if categories is None:
            with open(dataset_dir / 'validate' / 'labels.json') as f:
                categories = [
                    cat['name'] for cat in json.load(f)['categories']
                ]
        self.categories = categories

    def _load_model(self):
        if self.model_type == 'detectron':
            from PhagoPred.segmentation.detectron_infer import load_model
            return load_model(self.model_dir)
        elif self.model_type == 'cellpose':
            from PhagoPred.segmentation.cellpose_infer import load_model
            return load_model(self.model_dir)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type!r}")

    def _infer(self, im: np.ndarray, model) -> np.ndarray:
        """Dispatch to model-specific inference; always returns combined integer mask."""
        if self.model_type == 'detectron':
            from PhagoPred.segmentation.detectron_infer import seg_image
            im_rgb = np.stack([im] * 3, axis=-1) if im.ndim == 2 else im
            masks = seg_image(im_rgb, model_dir=self.model_dir, model=model)
            if masks is None:
                return np.zeros(im.shape[:2], dtype=np.int16)
            return mask_funcs.combine_masks(list(masks.values()))

        elif self.model_type == 'cellpose':
            from PhagoPred.segmentation.cellpose_infer import seg_image
            return seg_image(im, model=model)

    def eval(self) -> None:
        tools.remake_dir(self.eval_dir)
        model = self._load_model()
        coco_file = self.dataset_dir / 'validate' / 'labels.json'
        # val_images = sorted((self.dataset_dir / 'validate' / 'images').iterdir())
        val_images = (self.dataset_dir / 'validate' / 'images').glob('*.jpg')

        for im_path in tqdm(val_images,
                            desc=f'Evaluating [{self.model_type}]'):
            im = plt.imread(im_path)

            pred_mask = self._infer(im, model)
            true_masks = mask_funcs.coco_to_masks(coco_file=coco_file,
                                                  im_name=im_path)
            true_mask = mask_funcs.combine_masks(list(true_masks.values()))

            view = tools.show_segmentation(im, pred_mask, true_mask)
            plt.imsave(self.eval_dir / f'{im_path.stem}_view.png', view / 255)

            Image.fromarray(pred_mask.astype(np.int32), mode='I').save(
                self.eval_dir / f'{im_path.stem}_pred_mask.png')

            results = self.prec_recall_curve(true_mask, pred_mask)
            results.to_csv(self.eval_dir / f'{im_path.stem}_all_results.txt',
                           sep='\t')

        self.plot()

    def prec_recall_curve(
            self,
            true_mask: np.ndarray,
            pred_mask: np.ndarray,
            thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
    ) -> pd.DataFrame:
        import cellpose.metrics
        _, TPs, FPs, FNs = cellpose.metrics.average_precision(
            true_mask.astype(int), pred_mask.astype(int), threshold=thresholds)

        precision_denom = TPs + FPs
        recall_denom = TPs + FNs
        f1_denom = TPs + 0.5 * (FPs + FNs)
        no_truth_no_pred = (TPs == 0) & (FPs == 0) & (FNs == 0)

        precisions = np.where(
            no_truth_no_pred, 1.0,
            np.divide(TPs,
                      precision_denom,
                      out=np.zeros_like(TPs, dtype=float),
                      where=precision_denom != 0))
        recalls = np.where(
            no_truth_no_pred, 1.0,
            np.divide(TPs,
                      recall_denom,
                      out=np.zeros_like(TPs, dtype=float),
                      where=recall_denom != 0))
        F1s = np.where(
            no_truth_no_pred, 1.0,
            np.divide(TPs,
                      f1_denom,
                      out=np.zeros_like(TPs, dtype=float),
                      where=f1_denom != 0))

        return pd.DataFrame(
            {
                'Precision': precisions,
                'Recall': recalls,
                'F1': F1s
            },
            index=thresholds)

    def plot(self) -> None:
        plt.rcParams['font.family'] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        cmap = plt.cm.get_cmap('Set1')
        self.plot_category('all', axs, cmap(0))
        plt.tight_layout()
        plt.savefig(self.model_dir.parent / 'results.png')
        plt.close()

    def plot_category(self,
                      category: str,
                      axs,
                      colour,
                      label: str = None) -> None:
        if label is None:
            label = category
        results = [
            pd.read_csv(f, sep='\t', index_col=0)
            for f in self.eval_dir.glob(f'*_{category}_results.txt')
        ]
        if not results:
            return
        results = pd.concat(results, axis=0)
        percentiles = results.groupby(level=0).quantile([0.05, 0.5, 0.95
                                                         ]).unstack(level=1)
        thresholds = results.groupby(level=0).mean().index.values

        for ax, metric in zip(axs, ['Precision', 'Recall', 'F1']):
            p5 = percentiles[(metric, 0.05)]
            median = percentiles[(metric, 0.5)]
            p95 = percentiles[(metric, 0.95)]
            ax.plot(thresholds, median, color=colour, label=label)
            ax.fill_between(thresholds,
                            p5,
                            p95,
                            color=colour,
                            alpha=0.5,
                            edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
