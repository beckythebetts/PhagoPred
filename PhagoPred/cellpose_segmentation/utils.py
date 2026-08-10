from pathlib import Path
import os
import shutil

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

from PhagoPred.utils.mask_funcs import coco_to_masks


def detectron_to_cellpose(directory: Path) -> None:
    """Convert detectron2 style trainng data (images and coco_json) to cell pose (images and masks)."""
    images_folder = directory / 'images'
    labels_json = directory / 'labels.json'

    all_folder = directory / 'all'
    os.makedirs(all_folder, exist_ok=True)

    for im in images_folder.glob('*'):
        shutil.copy(im, all_folder / f"{im.stem}_im.png")
        mask = coco_to_masks(labels_json,
                             im.name)['Macrophage'].astype(np.uint16)
        mask = Image.fromarray(mask)
        mask.save(all_folder / f"{im.stem}_mask.png")


def compare_detectron_cellpose_prec_recall(dirs: dict[str, Path],
                                           save_path: Path) -> None:
    plt.rcParams["font.family"] = 'serif'
    cmap = plt.get_cmap('Set1')
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i, (label, dir) in enumerate(dirs.items()):
        results = []
        for split_dir in dir.glob('*model*'):
            if (split_dir / 'Evaluation').is_dir():
                # detectron2 model
                for file in (split_dir /
                             'Evaluation').glob('*all_results.txt'):
                    results.append(pd.read_csv(file, sep='\t', index_col=0))
            else:
                #cellpose model
                for file in (split_dir / 'validate').glob('*results.txt'):
                    results.append(pd.read_csv(file, sep='\t', index_col=0))
        results = pd.concat(results, axis=0)

        # Get percentiles
        percentiles = results.groupby(level=0).quantile([0.05, 0.5, 0.95
                                                         ]).unstack(level=1)
        metrics = results.columns.values
        thresholds = percentiles.index.values

        for ax, metric in zip(axs, metrics):
            p5 = percentiles[(metric, 0.05)]
            median = percentiles[(metric, 0.5)]
            p95 = percentiles[(metric, 0.95)]
            ax.plot(thresholds, median, color=cmap(i), label=label)
            ax.fill_between(thresholds,
                            p5,
                            p95,
                            color=cmap(i),
                            alpha=0.5,
                            edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel(metric.capitalize())
            ax.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    # detectron_to_cellpose(
    #     Path(
    #         "/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1_clahe/kfold"
    #     ))

    compare_detectron_cellpose_prec_recall(
        {
            '':
            Path(
                '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1_clahe_withrescale/kfold'
            ),
            # 'No Rescale':
            # Path(
            #     '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1_clahe/kfold'
            # )
            # 'Mask R-CNN':
            # Path(
            #     '/home/ubuntu/PhagoPred/PhagoPred/detectron_segmentation/models/bio_20x_thp1_clahe/kfold'
            # )
        },
        save_path=Path('temp') / 'seg_comapre.png')
