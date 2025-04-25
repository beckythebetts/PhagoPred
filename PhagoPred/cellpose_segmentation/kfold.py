from pathlib import Path
import numpy as np
import os
import shutil
import pandas as pd
import json
import matplotlib.pyplot as plt

from PhagoPred.cellpose_segmentation import train, eval
from PhagoPred.utils import tools, mask_funcs

class CellposeKfold:
    def __init__(self, directory):
        self.directory = directory
        self.ims = sorted([im for im in (self.directory / 'all').glob('*im*')])
        self.masks = sorted([mask for mask in (self.directory / 'all').glob('*mask*')])

    # def split_datasets(self):
    #     for test_pair in self.pairs:
    #         train_dir = self.directory / f'test_with_{test_pair[0].stem[:2]}' / 'train'
    #         val_dir = self.directory / f'test_with_{test_pair[0].stem[:2]}' / 'validate'
    #         train_dir.mkdir(parents=True)
    #         val_dir.mkdir()
    #         for item in test_pair:
    #             shutil.copy(item, val_dir)
    #         for train_pair in self.pairs:
    #             if train_pair != test_pair:
    #                 for item in train_pair:
    #                     shutil.copy(item, train_dir)

    def split_all(self, num_val=1):
        for i in range(len(self.ims)):
            training_data = self.directory / f'model_{i}'

            tools.remake_dir(training_data / 'train')
            tools.remake_dir(training_data / 'validate')

            val_ims = [self.ims[(i+x) % len(self.ims)] for x in range(num_val)]
            train_ims = [im for im in self.ims if im not in val_ims]

            # val_masks = [self.masks[(i+x) % len(self.masks)] for x in range(num_val)]
            # train_masks = [mask for mask in self.masks if im not in val_masks]


            for im, mask in zip(self.ims, self.masks):
                if im in val_ims:
                    shutil.copy(im, training_data / 'validate' / im.name)
                    shutil.copy(mask, training_data / 'validate' / mask.name)
                elif im in train_ims:
                    shutil.copy(im, training_data / 'train' / im.name)
                    shutil.copy(mask, training_data / 'train' / mask.name)

    def train_eval_datasets(self):
        for dataset_dir in self.directory.glob('*model*'):
            if dataset_dir.name != 'all':
                # train.cellpose_train(dataset_dir)
                eval.cellpose_eval(dataset_dir / 'validate', model=dataset_dir / 'models' / 'model')

    def get_results(self):
        results = []
        for dataset_dir in self.directory.glob('*model*'):
            if dataset_dir.name != 'all':
                for file in (dataset_dir / 'validate').iterdir():
                    if file.suffix == '.txt':
                        resultsi = pd.read_csv(file, sep='\t', index_col=0)
                        results.append(resultsi)
        results = pd.concat(results, axis=0)
        self.means, self.stds = results.groupby(level=0).mean(), results.groupby(level=0).std()
        self.means.to_csv(str(self.directory / 'results_means.txt'), sep='\t')
        self.stds.to_csv(str(self.directory / 'results_stds.txt'), sep='\t')

    def plot_results(self):
        metrics = self.means.columns.values
        thresholds = self.means.index.values
        print(thresholds)
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for ax, metric in zip(axs, metrics):
            ax.plot(thresholds, self.means[metric], color='red', label='Cellpose')
            ax.fill_between(thresholds, self.means[metric] - self.stds[metric],
                            self.means[metric] + self.stds[metric], color='red', alpha=0.5, edgecolor='none')
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric)
            ax.grid(True)
        plt.savefig(str(self.directory / 'results.png'))
        plt.clf()

    def plot_losses(self):
        train_losses = []
        test_losses = []
        file_names = [file / 'losses.txt' for file in self.directory.iterdir() if file.name != 'all']
        for dataset_dir in self.directory.iterdir():
            if dataset_dir.name != 'all' and dataset_dir.is_dir():
                with open(dataset_dir / 'losses.txt') as f:
                    losses = json.load(f)
                    train_losses.append(losses['Train Losses'])
                    test_losses.append(losses['Validation Losses'])
        train_losses = np.vstack(train_losses)
        test_losses = np.vstack(test_losses)
        train_means = np.mean(train_losses, axis=0)
        test_means = np.mean(test_losses, axis=0)
        train_stds = np.std(train_losses, axis=0)
        test_stds = np.std(test_losses, axis=0)
        iterations = np.arange(0, len(train_means))
        test_iterations, test_means, test_stds = np.array([[it, mean, std] for it, mean, std in zip(iterations, test_means, test_stds) if mean !=0]).transpose()
        plt.rcParams["font.family"] = 'serif'

        plt.plot(test_iterations, test_means, color='red', label='Validation loss')
        plt.fill_between(test_iterations, test_means-test_stds, test_means+test_stds, color='red', alpha=0.5, edgecolor='none')

        plt.plot(iterations, train_means, color='navy', label='Train loss')
        plt.fill_between(iterations, train_means - train_stds, train_means + train_stds, color='navy', alpha=0.5,
                         edgecolor='none')

        plt.grid()
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig(str(self.directory / 'av_loss_plot.png'))
        plt.clf()


def plot_training_data(training_data_values=[0, 2, 4]):
    directories = [Path('cellpose_Models') / f'kfold_manual_validation_{i}' for i in training_data_values]
    means = [pd.read_csv(directory / 'results_means.txt', sep='\t', index_col=0) for directory in directories]
    stds = [pd.read_csv(directory / 'results_stds.txt', sep='\t', index_col=0) for directory in directories]

    metrics = means[0].columns.values
    thresholds = means[0].index.values

    plt.rcParams["font.family"] = 'serif'
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axs, metrics):
        for mean, std, value in zip(means, stds, training_data_values):
            ax.plot(thresholds, mean[metric], label=f'{value} training images')
            ax.fill_between(thresholds, mean[metric]-std[metric], mean[metric]+std[metric], alpha=0.5, edgecolor=None)
            ax.grid(True)
            ax.set_xlabel('IOU Threshold')
            ax.set_ylabel(metric)
    plt.legend()
    plt.savefig(Path('cellpose_Models') / 'training_data_results.png')

    im_views = [plt.imread(directory / 'test_with_00' / 'validate' /'00_view.png')[200:800, 200:800] for directory in directories]
    side_by_side = np.concatenate(im_views, axis=1)
    plt.imsave(Path('cellpose_Models') / 'view_results.png', side_by_side)


def main():
    kfold = CellposeKfold(Path('PhagoPred') / 'cellpose_segmentation' / 'Models' / '20x_flir')
    # kfold.split_all()
    # kfold.train_eval_datasets()
    kfold.get_results()
    kfold.plot_results()
    kfold.plot_losses()
    plot_training_data()


if __name__ == '__main__':
    main()
