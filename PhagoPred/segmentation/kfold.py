from pathlib import Path
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from PhagoPred.utils import tools, mask_funcs
from PhagoPred.segmentation.evaluate import Evaluator


def make_coco_subset(orig_coco: Path, ims: list) -> dict:
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
            new_coco['images'].append({
                'id': len(new_coco['images']) + 1,
                'file_name': image['file_name'],
                'width': image['width'],
                'height': image['height'],
            })
            for annotation in orig_coco['annotations']:
                if annotation['image_id'] == image['id']:
                    new_annotation = annotation.copy()
                    new_annotation['image_id'] = len(new_coco['images'])
                    new_annotation['id'] = annotation_id
                    annotation_id += 1
                    new_coco['annotations'].append(new_annotation)
    return new_coco


class KFold:
    """
    K-fold cross-validation for instance segmentation, supporting detectron2 and cellpose.

    Expected input layout:
        directory/
            images/       <- all images
            labels.json   <- COCO-format annotations

    Each fold creates:
        directory/model_N/
            Training_Data/
                train/
                    images/            <- images + {stem}mask.png files
                    labels.json        <- COCO subset
                validate/   (same structure)
            Model/            <- model weights saved here after training
            Evaluation/       <- metrics and visualisations after eval
    """

    def __init__(self, directory: Path, model_type: str):
        """
        Args:
            directory:  root directory containing images/ and labels.json
            model_type: 'detectron' or 'cellpose'
        """
        self.directory = directory
        self.model_type = model_type
        self.ims = sorted((directory / 'images').iterdir())
        self.coco = directory / 'labels.json'
        with open(self.coco) as f:
            self.categories = [
                cat['name'] for cat in json.load(f)['categories']
            ]

    def split_all(self, num_val: int = 1) -> None:
        for i in range(int(len(self.ims) / num_val)):
            training_data = self.directory / f'model_{i}' / 'Training_Data'

            for split in ('train', 'validate'):
                tools.remake_dir(training_data / split / 'images')

            val_ims = [
                self.ims[((i * num_val) + x) % len(self.ims)]
                for x in range(num_val)
            ]
            train_ims = [im for im in self.ims if im not in val_ims]

            for split, split_ims in [('train', train_ims),
                                     ('validate', val_ims)]:
                split_dir = training_data / split
                coco_subset = make_coco_subset(self.coco, split_ims)
                with open(split_dir / 'labels.json', 'w') as f:
                    json.dump(coco_subset, f)
                for im_path in split_ims:
                    shutil.copy(im_path, split_dir / 'images' / im_path.name)
                self._create_mask_files(split_dir, split_ims)

    def _create_mask_files(self, split_dir: Path, split_ims: list) -> None:
        """
        Write a combined integer mask PNG alongside each image in split_dir/images/.
        Masks are rasterised from COCO annotations by combining all categories.
        Named {stem}mask.png — ignored by detectron, used by cellpose training.
        """
        coco_file = split_dir / 'labels.json'
        for im_path in split_ims:
            true_masks = mask_funcs.coco_to_masks(coco_file=coco_file,
                                                  im_name=split_dir /
                                                  'images' / im_path.name)
            combined = mask_funcs.combine_masks(list(true_masks.values()))
            Image.fromarray(combined.astype(np.uint16)).save(
                split_dir / 'images' / f'{im_path.stem}mask.png')

    def train(self) -> None:
        for fold_dir in sorted(self.directory.glob('model_*')):
            if self.model_type == 'detectron':
                self._train_detectron(fold_dir)
            elif self.model_type == 'cellpose':
                self._train_cellpose(fold_dir)

    def _train_detectron(self, fold_dir: Path) -> None:
        import sys
        sys.path.insert(0, 'detectron2')
        from PhagoPred.detectron_segmentation.train import train as det_train
        from PhagoPred.detectron_segmentation.kfold import unregister_coco_instances
        det_train(directory=fold_dir)
        unregister_coco_instances('my_dataset_train')
        unregister_coco_instances('my_dataset_val')

    def _train_cellpose(self, fold_dir: Path) -> None:
        from cellpose import models, core, train as cp_train
        training_data = fold_dir / 'Training_Data'

        def _load_split(images_dir: Path):
            image_paths = sorted(p for p in images_dir.iterdir()
                                 if 'mask' not in p.stem)
            mask_paths = sorted(images_dir.glob('*mask*'))
            imgs = [plt.imread(p) for p in image_paths]
            lbls = [np.array(Image.open(p)) for p in mask_paths]
            return imgs, lbls

        images, labels = _load_split(training_data / 'train' / 'images')
        test_images, test_labels = _load_split(training_data / 'validate' /
                                               'images')
        use_GPU = core.use_gpu()
        model = models.CellposeModel(gpu=use_GPU, pretrained_model='cpsam')
        model_dir = fold_dir / 'Model'
        model_dir.mkdir(exist_ok=True)
        model_path, train_losses, test_losses = cp_train.train_seg(
            model.net,
            train_data=images,
            train_labels=labels,
            normalize=True,
            test_data=test_images,
            test_labels=test_labels,
            learning_rate=1e-5,
            n_epochs=100,
            save_path=str(model_dir),
            model_name='model',
            batch_size=2)
        losses = {
            'Train Losses': np.array(train_losses).tolist(),
            'Validation Losses': np.array(test_losses).tolist(),
        }
        with open(fold_dir / 'losses.json', 'w') as f:
            json.dump(losses, f)
        self._plot_losses(fold_dir, losses)

    def save_pretrained_cellpose(self) -> None:
        from cellpose import models, core
        import torch
        use_GPU = core.use_gpu()
        model = models.CellposeModel(gpu=use_GPU, pretrained_model='cpsam')
        for fold_dir in sorted(self.directory.glob('model_*')):
            model_dir = fold_dir / 'Model'
            model_dir.mkdir(exist_ok=True)
            torch.save(model.net.state_dict(), model_dir / 'model')

    def _plot_losses(self, fold_dir: Path, losses: dict) -> None:
        train = np.array(losses['Train Losses'])
        val = np.array(losses['Validation Losses'])
        epochs = np.arange(len(train))
        val_epochs = epochs[val != 0]
        val_values = val[val != 0]
        plt.rcParams['font.family'] = 'serif'
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train, color='navy', alpha=0.7, label='Train')
        plt.scatter(val_epochs,
                    val_values,
                    color='red',
                    zorder=5,
                    label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fold_dir / 'loss_plot.png', dpi=150)
        plt.close()

    def eval(self) -> None:
        for fold_dir in sorted(self.directory.glob('model_*')):
            evaluator = Evaluator(dataset_dir=fold_dir / 'Training_Data',
                                  model_dir=fold_dir / 'Model',
                                  model_type=self.model_type,
                                  categories=self.categories)
            evaluator.eval()

    def plot_metrics_vs_iou(self,
                            category: str = 'all',
                            save_as: Path = None) -> None:
        plt.rcParams['font.family'] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        _plot_metrics(self.directory, axs, category, colour='b')
        for ax, metric in zip(axs, ['Precision', 'Recall', 'F1']):
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1.05)
            ax.grid(True)
        axs[0].legend()
        plt.tight_layout()
        out = save_as or self.directory / f'metrics_vs_iou_{category}.png'
        plt.savefig(out)
        plt.close()
        print(f"Saved to {out}")


def _plot_metrics(kfold_dir: Path,
                  axs: tuple[plt.Axes],
                  category: str = 'all',
                  colour='k',
                  label: str = '') -> None:
    results = []
    for fold_dir in sorted(kfold_dir.glob('model_*')):
        eval_dir = fold_dir / 'Evaluation'
        if not eval_dir.exists():
            continue
        for f in eval_dir.glob(f'*_{category}_results.txt'):
            results.append(pd.read_csv(f, sep='\t', index_col=0))
    if not results:
        print(f"No results found for category '{category}'.")
        return
    results = pd.concat(results, axis=0)
    means = results.groupby(level=0).mean()
    stds = results.groupby(level=0).std()
    thresholds = means.index.values

    # plt.rcParams['font.family'] = 'serif'
    # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axs, ['Precision', 'Recall', 'F1']):
        ax.plot(thresholds, means[metric], color=colour, label=label)
        ax.fill_between(
            thresholds,
            means[metric] - stds[metric],
            means[metric] + stds[metric],
            color=colour,
            alpha=0.3,
            edgecolor='none',
            # label=f'{label} ±1 std',
        )


def plot_comparison(kfold_dirs: list[Path], labels: list[str],
                    save_path: Path) -> None:
    plt.rcParams['font.family'] = 'serif'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    cmap = plt.get_cmap('Set1')
    for i, (kfold_dir, label) in enumerate(zip(kfold_dirs, labels)):
        _plot_metrics(kfold_dir,
                      axs,
                      category='all',
                      colour=cmap(i),
                      label=label)
    for ax, metric in zip(axs, ['Precision', 'Recall', 'F1']):
        ax.set_xlabel('IOU Threshold')
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.grid(True)
    axs[0].legend()

    plt.tight_layout()
    plt.savefig(save_path)


def main():
    # kfold = KFold(
    #     directory=Path(
    #         'PhagoPred/cellpose_segmentation/Models/bio_20x_thp1/kfold_base'),
    #     model_type='cellpose',  # or 'cellpose'
    # )
    # kfold.split_all()
    # # kfold.train()
    # kfold.save_pretrained_cellpose()
    # kfold.eval()
    # kfold.plot_metrics_vs_iou(category='all')

    plot_comparison(
        [
            Path(
                '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1/kfold'
            ),
            Path(
                '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1/kfold_base'
            ),
            Path(
                '/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/bio_20x_thp1/kfold'
            )
        ],
        ['Cellpose finetuned', 'Cellpose base', 'Detectron Mask R-CNN'],
        save_path=Path('/home/ubuntu/PhagoPred/temp/model_comparison.png'),
    )


if __name__ == '__main__':
    main()
