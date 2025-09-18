import h5py
import imagej
import numpy as np
import torch
import sys
import time
from skimage.segmentation import find_boundaries
from skimage.transform import resize
import skimage.measure
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import argparse
from texttable import Texttable
import napari
import io
from PIL import Image
from matplotlib.colors import ListedColormap
from napari.utils.colormaps import colormap_utils
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from qtpy.QtWidgets import QWidget, QVBoxLayout

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
        # phase_data = tools.get_images(f, 'Phase', first_frame, last_frame)
        # epi_data = tools.get_images(f, 'Epi', first_frame, last_frame)
        phase_data = f['Images']['Phase'][first_frame:last_frame]
        epi_data = f['Images']['Epi'][first_frame:last_frame]
    merged_im = np.stack((phase_data, phase_data, phase_data), axis=-1).astype(np.float32)
    # for i, im in enumerate(merged_im):
    #     cv2.putText(im, text=f'frame: {i+first_frame} | time: {(i+first_frame)*SETTINGS.TIME_STEP}/s', org=(20, SETTINGS.IMAGE_SIZE[1]-20), fontFace=cv2.FONT_HERSHEY_COMPLEX, 
    #                 fontScale=3, color=(255, 255, 255), thickness=3)
 
    merged_im[:, :, :, 0] += epi_data
    merged_im = np.clip(merged_im, a_min=0, a_max=255).astype(np.uint8)

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

def show_cell_images(cell_idx, first_frame=None, last_frame=None, frame_size=300):
    hdf5_file = SETTINGS.DATASET
    with h5py.File(hdf5_file, 'r') as f:
        first, last = tools.get_cell_end_frames(cell_idx, f)
        if first_frame is None:
            first_frame = first
        if last_frame is None:
            last_frame = last

        print(f'\nSHOWING CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}')

        n_frames = last_frame - first_frame
        phase_data = np.empty((n_frames, frame_size, frame_size))
        epi_data = np.empty((n_frames, frame_size, frame_size))
        mask = np.empty((n_frames, frame_size, frame_size), dtype=np.int32)

        x_centres = f['Cells']['Phase']['X'][first_frame:last_frame, cell_idx]
        y_centres = f['Cells']['Phase']['Y'][first_frame:last_frame, cell_idx]
        x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(y_centres)

        phase_stack = f['Images']['Phase'][first_frame:last_frame]
        epi_stack = f['Images']['Epi'][first_frame:last_frame]
        mask_stack = f['Segmentations']['Phase'][first_frame:last_frame]

        for idx in range(n_frames):
            xmin, xmax, ymin, ymax = mask_funcs.get_crop_indices(
                (y_centres[idx], x_centres[idx]), frame_size, SETTINGS.IMAGE_SIZE
            )
            phase_data[idx] = phase_stack[idx][ymin:ymax, xmin:xmax]
            epi_data[idx] = epi_stack[idx][ymin:ymax, xmin:xmax]
            mask[idx] = mask_stack[idx][ymin:ymax, xmin:xmax]

        cell_mask = (mask == cell_idx)

        if not cell_mask.any():
            raise Exception(f'Cell of index {cell_idx} not found')
        
        epi_data = tools.threshold_image(epi_data)
        # contours = skimage.measure.find_contours(cell_mask, level=0.5)
        cell_outline = mask_funcs.mask_outline(torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()

        # Threshold epi_data (assuming your existing tool does that)
        epi_data = tools.threshold_image(epi_data)

    viewer = napari.Viewer()
    viewer.add_image(phase_data, name='Phase')
    viewer.add_image(epi_data, name='Epi', blending='additive', colormap='red', opacity=0.5)

    viewer.add_image((cell_outline*255).astype(np.uint8), name='Outline', colormap='yellow', opacity=0.8, blending='additive')
    plot_widget = get_feature_plot_widget(cell_idx, first_frame, last_frame)
    viewer.window.add_dock_widget(plot_widget, area='right', name='Feature_plot')
    
    napari.run()

def get_feature_plot_widget(cell_idx, first_frame=0, last_frame=50, target_width=20):
    plt.rcParams["font.family"] = 'serif'
    
    feature_names=[
            'Area',
            # 'Circularity',
            # 'Perimeter',
            # 'Displacement',
            # 'Mode 0',
            # 'Mode 1',
            # 'Mode 2',
            # 'Mode 3',
            # 'Speed',
            # 'Phagocytes within 100 pixels',
            # 'Phagocytes within 250 pixels',
            # 'Phagocytes within 500 ',
            'X',
            'Y',
            'CellDeath'
    ]
    with h5py.File(SETTINGS.DATASET, 'r') as f:

        first, last = tools.get_cell_end_frames(cell_idx, f)
        if first_frame is None:
            first_frame = first
        if last_frame is None:
            last_frame = last

        print(f'\nPLOTTING FEATURES CELL: {cell_idx}, FRAMES: {first_frame} to {last_frame}\n')

        fig, axs = plt.subplots(len(feature_names), sharex=True, figsize=(10, 10))
        for i, feature_name in enumerate(feature_names):
            data = f['Cells']['Phase'][feature_name][first_frame:last_frame, cell_idx]
            axs[i].plot(range(first_frame,last_frame), data, color='k')
            # axs[i].set(ylabel=feature_name)
            axs[i].set_ylabel(feature_name, rotation=45, labelpad=20)
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame-1)

        fig.suptitle(f'Cell {cell_idx:04}')
        axs[-1].set(xlabel='frames')
        # plt.show()
    fig.tight_layout()

    # Wrap figure in a QWidget for Napari
    canvas = FigureCanvas(fig)
    widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(canvas)
    widget.setLayout(layout)

    return widget

def show_cell(cell_idx, first_frame=None, last_frame=None):
    # open both imagej of cell and feature plot
    show_cell_images(cell_idx, first_frame, last_frame)
    # multiprocessing.set_start_method('spawn')
    # imagej_thread = multiprocessing.Process(target=show_cell_images, args=(cell_idx, first_frame, last_frame))
    # plt_thread = multiprocessing.Process(target=show_feature_plot, args=(cell_idx, first_frame, last_frame))

    # imagej_thread.start()
    # plt_thread.start()

    # imagej_thread.join()
    # plt_thread.join()


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
    subparsers = parser.add_subparsers(dest='command', help='Function to execute')

    parser_show_raw = subparsers.add_parser('show_raw_images')
    parser_show_raw.add_argument('first_frame', type=int)
    parser_show_raw.add_argument('last_frame', type=int)

    parser_show_tracked = subparsers.add_parser('show_tracked_images')
    parser_show_tracked.add_argument('first_frame', type=int)
    parser_show_tracked.add_argument('last_frame', type=int)

    parser_show_cell = subparsers.add_parser('show_cell')
    parser_show_cell.add_argument('cell_idx', type=int)
    parser_show_cell.add_argument('--first_frame', type=int, default=None)
    parser_show_cell.add_argument('--last_frame', type=int, default=None)


    parser_print_phago = subparsers.add_parser('print_phago')

    args = parser.parse_args()

    if args.command == 'show_raw_images':
        show_raw_images(args.first_frame, args.last_frame)
    elif args.command == 'show_tracked_images':
        show_tracked_images(args.first_frame, args.last_frame)
    elif args.command == 'show_cell':
        # show_cell(args.cell_idx, a)
        show_cell(args.cell_idx, args.first_frame, args.last_frame)
    elif args.command == 'print_phago':
        print_phagocytosis()


if __name__ == '__main__':
    main()