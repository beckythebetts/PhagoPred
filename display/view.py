import h5py
import imagej
import scyjava
import numpy as np
import torch
import sys
import time
#from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import argparse

from PhagoPred.utils import mask_funcs
from PhagoPred import SETTINGS
#from jnius import autoclass

#hdf5_file = 'Datasets/04_short.h5'
#hdf5_file = 'Datasets/danhighres/dan10.h5'
hdf5_file = SETTINGS.DATASET
#ij = imagej.init('sc.fiji:fiji:2.1.0', mode='gui')
#scyjava.config.add_options('-Xmx2g')
ij = imagej.init('2.1.0', mode='interactive')
#ij = imagej.init(mode='interactive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def make_rgb(greyscale_im, axis=1):
#     return np.stack((greyscale_im, greyscale_im, greyscale_im), axis=axis)[:, np.newaxis]

# def show_separate_channels(first_frame=0, last_frame=50):
#     print('\nSHOWING SEPARATE CHANNELS')
#     with h5py.File(hdf5_file, 'r') as f:
#         data = np.array([
#             [
#                 f['Images'][channel][frame][:]
#                 for channel in f['Images'].keys()
#             ]
#             for frame in f['Images']['Phase'].keys()
#         ], dtype='uint8')
#         ij_image = ij.py.to_java(data)
#         ij.ui().show('images', ij_image)
#     time.sleep(99999)

# def show_merged_channels_old(first_frame=0, last_frame=50):
#     print('\nSHOWING MERGED IMAGES')
#     with h5py.File(hdf5_file, 'r') as f:
#         phase_data = np.array([f['Images']['Phase'][frame][:]
#                                for frame in list(f['Images']['Phase'].keys())[first_frame:last_frame]], dtype='uint8')
#         epi_data = np.array([f['Segmentations']['Epi'][frame][:]
#                                for frame in list(f['Images']['Epi'].keys())[first_frame:last_frame]], dtype='uint8')
#
#     epi_data[epi_data > 0] = 255
#     epi_channel = make_rgb(epi_data)
#     epi_channel[:,:,1:3] = 0
#     merged_data = (0.75*(make_rgb(phase_data).astype(np.float32) + 0.25*epi_channel.astype(np.float32))).astype(np.uint8)
#     # only merge red channel (0)
#     # merged_im = make_rgb(phase_data)
#     # merged_im[:, :, 0] = (phase_data).astype(np.float32)+epi_channel[:, :, 0]
#     merged_image = ij.py.to_dataset(merged_data, dim_order=['time', 'z', 'ch', 'row', 'col'])
#     ij.ui().show(merged_image)
#     ij.py.run_macro(macro='run("Make Composite")')
#     time.sleep(99999)

def show_raw_images(first_frame=0, last_frame=50):
    print('\nSHOWING MERGED IMAGES')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in list(f['Images']['Phase'].keys())[first_frame:last_frame]], dtype='uint8')
        epi_data = np.array([f['Images']['Epi'][frame][:]
                               for frame in list(f['Images']['Epi'].keys())[first_frame:last_frame]], dtype='uint8')
    print(epi_data.shape, phase_data.shape)
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=1)
    print(merged_im.shape)
    merged_im[:, 0][epi_data > SETTINGS.THRESHOLD] = epi_data[epi_data > SETTINGS.THRESHOLD]
    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)
# def show_tracked_images_old():
#     print('\nPREPARING TRACKED IMAGES\n')
#     with h5py.File(hdf5_file, 'r') as f:
#         phase_data = np.array([f['Images']['Phase'][frame][:]
#                                for frame in f['Images']['Phase'].keys()][:3], dtype='uint8')
#         segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
#                                       for frame in list(f['Segmentations']['Phase'].keys())][:3], dtype='int16')
#     max_cell_index=np.max(segmentation_data)
#     colour_dict = {cell_index: torch.tensor(np.random.uniform(0, (2 ** 8) - 1, size=3).astype('uint8')).to(device) for
#                    cell_index in np.arange(1, max_cell_index + 1)}
#     rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
#     tracked = np.zeros(rgb_phase.shape)
#     for i, (phase_image, segmentation) in enumerate(
#             zip(torch.tensor(rgb_phase).to(device), torch.tensor(segmentation_data).to(device))):
#         sys.stdout.write(
#             f'\rFrame {i + 1}')
#         sys.stdout.flush()
#         for cell_index in torch.unique(segmentation)[1:]:
#             outline = mask_funcs.mask_outline(torch.where(segmentation == cell_index.item(), 1, 0), thickness=3)
#             phase_image[outline] = colour_dict[cell_index.item()]
#
#         tracked[i] = phase_image.cpu().numpy()
#     tracked_image = ij.py.to_dataset(tracked, dim_order=['time', 'row', 'col', 'ch'])
#     ij.ui().show(tracked_image)
#     ij.py.run_macro(macro='run("Make Composite")')
#     time.sleep(99999)


def show_tracked_images(first_frame=0, last_frame=50):
    print('\nPREPARING TRACKED IMAGES\n')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = np.array([f['Images']['Phase'][frame][:]
                               for frame in f['Images']['Phase'].keys()][first_frame:last_frame], dtype='uint8')
        segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
                                      for frame in list(f['Segmentations']['Phase'].keys())][first_frame:last_frame], dtype='int16')
    max_cell_index = np.max(segmentation_data)
    LUT = torch.randint(low=10, high=255, size=(max_cell_index+1, 3)).to(device)
    LUT[0] = torch.tensor([0, 0, 0]).to(device)
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image, segmentation) in enumerate(
            zip(rgb_phase, segmentation_data)):
        segmentation = torch.tensor(segmentation).to(device)
        phase_image = torch.tensor(phase_image).to(device).int()
        sys.stdout.write(
            f'\rFrame {i + 1}')
        sys.stdout.flush()
        outlines = mask_funcs.mask_outlines(segmentation, thickness=1).int()
        outlines = LUT[outlines]
        phase_image =  (torch.where(outlines>0, outlines, phase_image))
        phase_image = phase_image.cpu().numpy().astype(np.uint8)
        tracked[i] = phase_image
    tracked_image = ij.py.to_dataset(tracked, dim_order=['t', 'row', 'col', 'ch'])
    ij.ui().show(tracked_image)
    ij.py.run_macro(macro='run("Make Composite")')
    #ij.py.run_macro(macro='run("8-bit")')
    time.sleep(99999)


def show_cell_images(cell_idx, first_frame=0, last_frame=50, frame_size=150):
    print(f'\nSHOWING CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}')
    phase_data = np.empty((last_frame-first_frame, frame_size, frame_size))
    epi_data = np.empty((last_frame-first_frame, frame_size, frame_size))
    mask_data = np.empty((last_frame-first_frame, frame_size, frame_size))
    with h5py.File(hdf5_file, 'r') as f:
        for idx, frame in enumerate(range(first_frame, last_frame)):
            xcentre = np.nan
            framei = frame
            while np.isnan(xcentre):
                xcentre, ycentre = f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['xcentre'], f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['ycentre']
                framei -= 1
            ymin, ymax, xmin, xmax = mask_funcs.get_crop_indices((ycentre, xcentre), frame_size, SETTINGS.IMAGE_SIZE)
            phase_data[idx] = np.array(f['Images']['Phase'][f'{int(frame):04}'])[xmin:xmax, ymin:ymax]
            epi_data[idx] = f['Images']['Epi'][f'{int(frame):04}'][xmin:xmax, ymin:ymax]
            mask_data[idx] = f['Segmentations']['Phase'][f'{int(frame):04}'][xmin:xmax, ymin:ymax]
        # xcentres = np.array([f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['xcentre'] for framei in
        #                     range(first_frame, last_frame)])
        # ycentres = np.array([f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][int(framei)]['ycentre'] for framei in
        #                     range(first_frame, last_frame)])
        # xmins, xmaxs, ymins, ymaxs = (xcentres-frame_size/2).astype(int), (xcentres+frame_size/2).astype(int), (ycentres-frame_size/2).astype(int), (ycentres+frame_size/2).astype(int)
        #
        # phase_data = np.array([f['Images']['Phase'][frame][xmin:xmax,ymin:ymax]
        #                        for frame, xmin, xmax, ymin, ymax in zip(list(f['Images']['Phase'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')
        # epi_data = np.array([f['Images']['Epi'][frame][xmin:xmax,ymin:ymax]
        #                      for frame, xmin, xmax, ymin, ymax in zip(list(f['Images']['Epi'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')
        # mask_data = np.array([f['Segmentations']['Phase'][frame][xmin:xmax,ymin:ymax]
        #                        for frame, xmin, xmax, ymin, ymax in zip(list(f['Segmentations']['Phase'].keys())[first_frame:last_frame], xmins, xmaxs, ymins, ymaxs)], dtype='uint8')
    cell_mask = (mask_data == cell_idx)
    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')
    cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=1).cpu().numpy()
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=1)
    merged_im[:, 0][epi_data > SETTINGS.THRESHOLD] = epi_data[epi_data > SETTINGS.THRESHOLD]
    merged_im[:, 0][cell_outline] = 255
    merged_im[:, 1][cell_outline] = 255
    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)

def show_feature_plot(cell_idx, first_frame=0, last_frame=50):
    plt.rcParams["font.family"] = 'serif'
    print(f'\nPLOTTING FEATURES CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}\n')
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        data = pd.DataFrame(f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][first_frame:last_frame])
        columns = [0, 1, 2, 3, 4, 7]
        fig, axs = plt.subplots(len(columns), sharex=True, figsize=(10, 10))
        for i, col in enumerate(columns):
            axs[i].plot(range(first_frame,last_frame), data.iloc[:, col], color='k')
            axs[i].set(ylabel=data.columns.values.tolist()[col])
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame-1)

        fig.suptitle(f'Cell{cell_idx:04}')
        axs[-1].set(xlabel='frames')
        plt.show()


def show_cell(cell_idx, first_frame=0, last_frame=50):
    # open both imagej of cell and feature plot
    multiprocessing.set_start_method('spawn')
    imagej_thread = multiprocessing.Process(target=show_cell_images, args=(cell_idx, first_frame, last_frame))
    plt_thread = multiprocessing.Process(target=show_feature_plot, args=(cell_idx, first_frame, last_frame))

    imagej_thread.start()
    plt_thread.start()

    imagej_thread.join()
    plt_thread.join()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='Funciton to execute')

    parser_show_raw = subparsers.add_parser('show_raw_images')
    parser_show_raw.add_argument('first_frame', type=int)
    parser_show_raw.add_argument('last_frame', type=int)

    parser_show_tracked = subparsers.add_parser('show_tracked_images')
    parser_show_tracked.add_argument('first_frame', type=int)
    parser_show_tracked.add_argument('last_frame', type=int)

    parser_show_cell = subparsers.add_parser('show_cell')
    parser_show_cell.add_argument('cell_idx', type=int)
    parser_show_cell.add_argument('first_frame', type=int)
    parser_show_cell.add_argument('last_frame', type=int)

    args = parser.parse_args()

    if args.command == 'show_raw_images':
        show_raw_images(args.first_frame, args.last_frame)
    elif args.command == 'show_tracked_images':
        show_tracked_images(args.first_frame, args.last_frame)
    elif args.command == 'show_cell':
        show_cell(args.cell_idx, args.first_frame, args.last_frame)


if __name__ == '__main__':
    main()