from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import torch

from PhagoPred.utils import tools
from PhagoPred.unet_segmentation import train, eval

class KFold:

    def __init__(self, directory):
        self.directory = directory
        self.ims = sorted([im for im in (self.directory / 'images').iterdir()])
        self.masks = sorted([im for im in (self.directory / 'masks').iterdir()])

    def split_all(self, num_val=4):
        for i in range(int(len(self.ims)/num_val)):
            training_data = self.directory / f'model_{i}' / 'Training_Data'

            tools.remake_dir(training_data / 'train' / 'images')
            tools.remake_dir(training_data / 'validate' / 'images')
            tools.remake_dir(training_data / 'train' / 'masks')
            tools.remake_dir(training_data / 'validate' / 'masks')

            val_ims = [self.ims[((i*num_val)+x) % len(self.ims)] for x in range(num_val)]
            train_ims = [im for im in self.ims if im not in val_ims]

            for im in self.ims:
                if im in val_ims:
                    shutil.copy(im, training_data / 'validate' / 'images' / im.name)
                    shutil.copy(self.directory / 'masks' / im.name, training_data / 'validate' / 'masks' / im.name)

                elif im in train_ims:
                    shutil.copy(im, training_data / 'train' / 'images' / im.name)
                    shutil.copy(self.directory / 'masks' / im.name, training_data / 'train' / 'masks' / im.name)

    def train_and_eval(self):
        for file in self.directory.glob('*model*'):
            # train.train(dir=file)
            eval.eval(dir=file)

    # def plot_loss(self):
    #     for file in self.directory.glob('*model*'):
    #         losses = pd.read_csv(file / 'training_losses.txt', sep='\t', header=0)
    #         plt.rcParams["font.family"] = 'serif'
    #         plt.scatter(losses['Epoch'], losses['Training Loss'], color='navy')
    #         plt.scatter(
    #             losses['Epoch'], losses['Validation Loss'], color='red')
    #         plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    #         plt.savefig(file / 'loss_plot.png')
    #         plt.clf()

    # def eval(self):
    #     for file in self.directory.glob('*model*'):
    #         model = torch.load(file / 'model.pth')
    #         model.eval()
    #         for im in file / 'Training_Data' / 'validate' / 'images'


def main():
    kfold = KFold(Path('PhagoPred') / 'unet_segmentation' / 'models' / '20x_flir_8')
    # kfold.split_all()
    kfold.train_and_eval()
    # kfold.plot_loss()

if __name__ == '__main__':
    main()