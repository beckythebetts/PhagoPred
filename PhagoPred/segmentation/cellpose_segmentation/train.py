import json
from pathlib import Path

from cellpose import io, models, core, train
import matplotlib.pyplot as plt
import numpy as np

from PhagoPred import SETTINGS
from PhagoPred.utils import mask_funcs


def cellpose_train(directory: Path):
    use_GPU = core.use_gpu()
    io.logger_setup()
    print(f'Loading train data from {directory / "train"}')
    images, labels, image_names, test_images, test_labels, image_names_test = io.load_train_test_data(
        str(directory / 'train'),
        str(directory / 'validate'),
        image_filter='im',
        mask_filter='mask')
    model = models.CellposeModel(gpu=use_GPU, pretrained_model='cpsam')
    # model = models.CellposeModel(gpu=use_GPU, diam_mean=mask_funcs.calculate_mean_diameter(directory / 'train'))

    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        # channels=[0, 0],
        normalize=True,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=0.1,
        SGD=True,
        learning_rate=1e-5,
        n_epochs=100,
        save_path=str(directory),
        model_name='model',
        batch_size=2,
        rescale=True)

    losses_dict = {
        'Train Losses': train_losses.tolist(),
        'Validation Losses': test_losses.tolist()
    }
    with open(str(directory / 'losses.txt'), 'w') as f:
        json.dump(losses_dict, f)
    epochs = np.arange(0, len(train_losses))
    plt.rcParams["font.family"] = 'serif'
    plt.scatter(epochs, train_losses, color='navy')
    validation_epochs, validation_losses = np.array(
        [[epoch, validation] for epoch, validation in zip(epochs, test_losses)
         if validation != 0]).transpose()
    plt.scatter(validation_epochs, validation_losses, color='red')
    plt.legend(['Train Losses', 'Validation Losses'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig(directory / 'loss_plot.png')
    plt.clf()


# def detectron_to_cellpose_dir_structure(dir_path: Path):
#     for im in (dir_path / 'images').iterdir():
#         mask = mask_funcs.convert_coco_file


def main():

    model_directory = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/cellpose_segmentation/Models/bio_20x_thp1_clahe_withrescale'
    )
    # mask_funcs.convert_coco_file(model_directory / 'train' / 'labels.json',
    #                              model_directory / 'train' / 'images',
    #                              model_directory / 'train' / 'data')
    # mask_funcs.convert_coco_file(model_directory / 'validate' / 'labels.json',
    #                              model_directory / 'validate' / 'images',
    #                              model_directory / 'validate' / 'data')
    cellpose_train(model_directory)


if __name__ == '__main__':
    main()
