import skimage.restoration
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import cv2
from scipy.ndimage import label
from skimage import restoration, exposure, filters, morphology, segmentation
import sys
import numpy as np
import h5py

from PhagoPred import SETTINGS


def test_chan_vese():
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        test_image = f['Images']['Epi'][list(f['Images']['Epi'].keys())[0]][...]
        mask = segmentation.chan_vese(test_image, mu=0.1, lambda1=1.0, lambda2=5.0)
        fig = plt.figure()
        grid = ImageGrid(fig, (0, 0, 1, 1), nrows_ncols=(1, 2))
        for ax, im in zip(grid, (test_image, mask)):
            ax.matshow(im)
            ax.axis('off')
        # plt.imshow(test_image)
        plt.show()


def test_watershed():
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        test_image = f['Images']['Epi'][list(f['Images']['Epi'].keys())[0]][...]
        mask = segmentation.watershed(test_image)
        fig = plt.figure()
        grid = ImageGrid(fig, (0, 0, 1, 1), nrows_ncols=(1, 2))
        for ax, im in zip(grid, (test_image, mask)):
            ax.matshow(im)
            ax.axis('off')
        # plt.imshow(test_image)
        plt.show()


def test_filter_and_threshold(test_threshold_value, psfsigma=1.5):
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        test_image = f['Images']['Epi'][list(f['Images']['Epi'].keys())[0]][...]
        #smoothed_image = filters.gaussian(test_image, sigma=3)
        #filtered_image = bilateral_filter.apply_bilateral_filter(test_image, iterations, d, sigmaColour, sigmaSpace)
        psf = np.zeros((21, 21))
        psf[10, 10] = 1
        psf = skimage.filters.gaussian(psf, sigma=psfsigma)
        psf = (psf/np.sum(psf))*255
        filtered_image, _ = restoration.unsupervised_wiener(test_image, psf)
        filtered_image = skimage.exposure.rescale_intensity(filtered_image, in_range=(0, 1), out_range=(0, 255))
        filtered_image = filters.gaussian(test_image, sigma=2)
        mask = np.where(filtered_image > test_threshold_value, 1, 0)
        print(mask)
        thresholded_image = np.stack((test_image, test_image, test_image), axis=-1)
        thresholded_image[:,:,1] = np.where(mask==1, 255, thresholded_image[:,:,1])

        fig = plt.figure()
        grid = ImageGrid(fig, (0,0,1,1), nrows_ncols=(2, 2))
        for ax, im in zip(grid, (test_image, np.where(test_image>test_threshold_value, 1, 0), filtered_image, mask)):
            ax.matshow(im)
            ax.axis('off')
        # plt.imshow(test_image)
        plt.show()
        # plt.matshow(mask)
        # plt.show()


def apply_threshold(threshold=SETTINGS.THRESHOLD):
    print('\n--------------------\nTHRESHOLDING - ', SETTINGS.CLASSES['epi'], '\n--------------------')
    with h5py.File(SETTINGS.DATASET, 'r+') as f:

        num_frames = f['Images'].attrs['Number of frames']
        for i, frame in enumerate(f['Images']['Epi']):
            sys.stdout.write(
                f'\rAdding frame {i + 1} / {num_frames}')
            sys.stdout.flush()
            image = f['Images']['Epi'][frame][...]
            image = np.where(image > threshold, 1, 0)
            image, num_objects = label(image)
            f.create_dataset(f'Segmentations/Epi/{frame}', dtype='i2', data=image)
            f['Segmentations']['Epi'].attrs['Threshold'] = threshold


def main():
    #test_watershed()
    #test_chan_vese()
    #test_filter_and_threshold(250, 20, -1, 20, 20)
    #test_filter_and_threshold(50)
    apply_threshold()

if __name__ == '__main__':
    main()