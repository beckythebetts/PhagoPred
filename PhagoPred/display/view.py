import h5py
import imagej
import numpy as np
import torch
import sys
import time
from skimage.segmentation import find_boundaries
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import argparse
from texttable import Texttable

from PhagoPred.utils import mask_funcs, tools
from PhagoPred import SETTINGS

hdf5_file = SETTINGS.DATASET

ij = imagej.init('2.1.0', mode='interactive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show_raw_images(first_frame=0, last_frame=50):
    """
    Displays in imagej epi_images (only pixels where pixel_value>epi_threshold) overlayed in red on phase images.

    Args:
        first_frame(int): first frame of dequence to display
        last_frame(int): last frame of sequence to display
    """
    print('\nSHOWING MERGED IMAGES')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = tools.get_images(f, 'Phase', first_frame, last_frame)
        epi_data = tools.get_images(f, 'Epi', first_frame, last_frame)
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=-1)
    for i, im in enumerate(merged_im):
        cv2.putText(im, text=f'frame: {i+first_frame} | time: {(i+first_frame)*SETTINGS.TIME_STEP}/s', org=(20, SETTINGS.IMAGE_SIZE[1]-20), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale=3, color=(255, 255, 255), thickness=3)
    merged_im[:, :, :, 0][epi_data > SETTINGS.THRESHOLD] = epi_data[epi_data > SETTINGS.THRESHOLD]
    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'row', 'col', 'ch'])
    ij.ui().show('ims', merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)


def show_tracked_images(first_frame=0, last_frame=50):
    print('\nPREPARING TRACKED IMAGES\n')
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = tools.get_images(f, 'Phase', first_frame, last_frame)
        # segmentation_data = np.array([f['Segmentations']['Phase'][frame][:]
        #                               for frame in list(f['Segmentations']['Phase'].keys())][first_frame:last_frame], dtype='int16')
        segmentation_data = tools.get_masks(f, 'Phase', first_frame, last_frame)
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
    for i, im in enumerate(tracked):
        cv2.putText(im, text=f'frame: {i+first_frame} | time: {(i+first_frame)*SETTINGS.TIME_STEP}/s', org=(20, SETTINGS.IMAGE_SIZE[1]-20), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale=3, color=(255, 255, 255), thickness=3)
    tracked_image = ij.py.to_dataset(tracked, dim_order=['t', 'row', 'col', 'ch'])
    ij.ui().show(tracked_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)

def show_cell_images(cell_idx, first_frame=0, last_frame=50, frame_size=150):
    print(f'\nSHOWING CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}')
    
    with h5py.File(hdf5_file, 'r') as f:
        x_centres, y_centres = tools.get_features_ds(f['Cells']['Phase'], 'x')[first_frame:last_frame, cell_idx], tools.get_features_ds(f['Cells']['Phase'], 'y')[first_frame:last_frame, cell_idx]
        # fill np.nan values with closest non np.values
        x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

        crop_idxs = mask_funcs.get_crop_indices_all(np.stack((x_centres, y_centres), axis=-1), side_length=frame_size, image_size=SETTINGS.IMAGE_SIZE)

        x_idxs = np.array([np.arange(xmin, xmax) for xmin, xmax in zip(crop_idxs[:,2], crop_idxs[:,3])])[:, :, np.newaxis]
        y_idxs = np.array([np.arange(ymin, ymax) for ymin, ymax in zip(crop_idxs[:,0], crop_idxs[:,1])])[:, np.newaxis, :]

        phase_data = tools.get_images(f, 'Phase', first_frame, last_frame)[np.arange(last_frame-first_frame)[:, None, None], x_idxs, y_idxs]
        epi_mask = (tools.get_masks(f, 'Epi', first_frame, last_frame)[np.arange(last_frame-first_frame)[:, None, None], x_idxs, y_idxs] > 0).astype(bool)
        phase_mask = tools.get_masks(f, 'Phase', first_frame, last_frame)[np.arange(last_frame-first_frame)[:, None, None], x_idxs, y_idxs]

        cell_mask = (phase_mask == cell_idx)
        
    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')
    cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=1).cpu().numpy()
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=1)

    merged_im[:, 0][epi_mask] = 255
    merged_im[:, 1][epi_mask] = 0
    merged_im[:, 2][epi_mask] = 0

    merged_im[:, 0][cell_outline] = 255
    merged_im[:, 1][cell_outline] = 255
    merged_im[:, 2][cell_outline] = 0

    merged_image = ij.py.to_dataset(merged_im, dim_order=['t', 'ch', 'row', 'col'])
    ij.ui().show(merged_image)
    ij.py.run_macro(macro='run("Make Composite")')
    time.sleep(99999)


def show_feature_plot(cell_idx, first_frame=0, last_frame=50):
    plt.rcParams["font.family"] = 'serif'
    print(f'\nPLOTTING FEATURES CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}\n')
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        data = f['Cells']['Phase'][first_frame:last_frame, cell_idx, 2:][:]
        feature_names = f['Cells']['Phase'].attrs['features'][2:]
        #data = pd.DataFrame(f['Features'][f'Cell{cell_idx:04}']['MorphologicalFeatures'][first_frame:last_frame])
        fig, axs = plt.subplots(len(feature_names), sharex=True, figsize=(10, 10))
        for i, feature_name in enumerate(feature_names):
            axs[i].plot(range(first_frame,last_frame), data[:, i], color='k')
            # axs[i].set(ylabel=feature_name)
            axs[i].set_ylabel(feature_name, rotation=45, labelpad=20)
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame-1)

        fig.suptitle(f'Cell {cell_idx:04}')
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

def print_phagocytosis():
    with h5py.File(SETTINGS.DATASET, 'r') as f:
        t = Texttable()
        t.header(['Phagocyte index', 'Pathogen index', 'First frame', 'Last frame'])
        for phago_event in f['Cells']['Phagocytosis'].keys():
            phago_event = f['Cells']['Phagocytosis'][phago_event]
            if phago_event[-1, 0]-phago_event[0, 0] > 20:
                t.add_row([int(phago_event.attrs['phagocyte_idx']), int(phago_event.attrs['pathogen_idx']), int(phago_event[0][0]), int(phago_event[-1][0])])
            #print(f"\n{phago_event.attrs['phagocyte_idx']}\t{phago_event.attrs['pathogen_idx']}\t{phago_event[0][0]}\t{phago_event[-1][0]}")
    print(t.draw())


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

    parser_print_phago = subparsers.add_parser('print_phago')

    args = parser.parse_args()

    if args.command == 'show_raw_images':
        show_raw_images(args.first_frame, args.last_frame)
    elif args.command == 'show_tracked_images':
        show_tracked_images(args.first_frame, args.last_frame)
    elif args.command == 'show_cell':
        show_cell(args.cell_idx, args.first_frame, args.last_frame)
    elif args.command == 'print_phago':
        print_phagocytosis()


if __name__ == '__main__':
    main()