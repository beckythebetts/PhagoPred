import h5py
import io
import textwrap
import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import os
from PIL import Image
import pycocotools
from tqdm import tqdm
from typing import Optional

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def save_masks(dir, first_frame=0, last_frame=50):
    hdf5_file = SETTINGS.DATASET
    with h5py.File(hdf5_file, 'r') as f:
        phase_masks = tools.get_masks(f, 'Phase', first_frame, last_frame)
    for i, im in enumerate(phase_masks):
        sys.stdout.write(f'\rFrame {i} / {len(phase_masks)}')
        sys.stdout.flush()
        plt.imsave(dir / f'{i:04}.png', im, cmap='gray')


def save_tracked_images(hdf5_file: Path,
                        save_dir: Path,
                        first_frame=0,
                        last_frame=50):
    """Save the specified frames as .jpegs, with each cell assigned a random colour for visual check of tracking
    """
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print('\nPREPARING TRACKED IMAGES\n')
    os.makedirs(save_dir, exist_ok=True)
    with h5py.File(hdf5_file, 'r') as f:
        phase_data = f['Images']['Phase'][first_frame:last_frame]
        segmentation_data = f['Segmentations']['Phase'][first_frame:last_frame]
    max_cell_index = np.max(segmentation_data)
    LUT = torch.randint(low=10,
                        high=255,
                        size=(max_cell_index + 1, 3),
                        dtype=torch.uint8).to(device)

    lut_for_neg1 = torch.tensor([[0, 0, 0]], device=device, dtype=torch.uint8)
    LUT = torch.cat([lut_for_neg1, LUT], dim=0)
    # LUT[0] = torch.tensor([0, 0, 0]).to(device)
    rgb_phase = np.stack((phase_data, phase_data, phase_data), axis=-1)
    # tracked = np.zeros(rgb_phase.shape)
    for i, (phase_image,
            segmentation) in tqdm(enumerate(zip(rgb_phase,
                                                segmentation_data))):
        segmentation = torch.tensor(segmentation).to(device)
        phase_image = torch.tensor(phase_image, dtype=torch.uint8).to(device)
        # sys.stdout.write(
        #     f'\rFrame {i + 1}')
        # sys.stdout.flush()
        outlines = mask_funcs.mask_outlines(segmentation,
                                            thickness=3).type(torch.int64)
        colour_outlines = (LUT[outlines]).type(torch.uint8)
        outlined_phase_image = (torch.where(
            outlines.unsqueeze(-1).expand_as(colour_outlines) > 0,
            colour_outlines, phase_image))
        outlined_phase_image = outlined_phase_image.cpu().numpy() / 256
        plt.imsave(save_dir / f'{i}.jpeg', outlined_phase_image)


def save_cell_images(hdf5_file,
                     dir,
                     cell_idx,
                     first_frame=0,
                     last_frame=10,
                     frame_size=300,
                     as_gif=False):
    """For a specified cell index, save {frame_size} x {frame_size} pixel images centred on the cell centre for the specified frames.
    Cell will be outlined in yellow, with fluorescence overlayed in red.
    """
    os.mkdir(dir)
    phase_data = np.empty((last_frame - first_frame, frame_size, frame_size))
    epi_data = np.empty((last_frame - first_frame, frame_size, frame_size))
    mask = np.empty((last_frame - first_frame, frame_size, frame_size))

    with h5py.File(hdf5_file, 'r') as f:
        x_centres, y_centres = f['Cells']['Phase']['X'][
            first_frame:last_frame,
            cell_idx], f['Cells']['Phase']['Y'][first_frame:last_frame,
                                                cell_idx]
        x_centres, y_centres = tools.fill_nans(x_centres), tools.fill_nans(
            y_centres)

        image_size = f['Segmentations']['Phase'][0].shape

        for idx, frame in tqdm(enumerate(range(first_frame, last_frame)),
                               total=last_frame - first_frame):
            ymin, ymax, xmin, xmax = mask_funcs.get_crop_indices(
                (y_centres[idx], x_centres[idx]), frame_size, image_size)
            phase_data[idx] = np.array(f['Images']['Phase'][frame])[xmin:xmax,
                                                                    ymin:ymax]
            # epi_data[idx] = (f['Images']['Epi'][frame][xmin:xmax, ymin:ymax])
            mask[idx] = f['Segmentations']['Phase'][frame][xmin:xmax,
                                                           ymin:ymax]
            cell_mask = (mask == cell_idx)

    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')

    # epi_data = tools.threshold_image(epi_data)
    cell_outline = mask_funcs.mask_outline(
        torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()
    epi_data = np.zeros_like(phase_data)
    merged_im = tools.overlay_red_on_gray(phase_data,
                                          epi_data,
                                          normalize=False)
    # merged_im = cell_outline[..., np.newaxis]
    # merged_im = np.concatenate([merged_im, merged_im, merged_im], axis=-1)
    merged_im[..., 0][cell_outline] = 255
    merged_im[..., 1][cell_outline] = 255

    if as_gif:
        ims = [Image.fromarray(img) for img in merged_im.astype(np.uint8)]
        ims[0].save(dir / f"cell{cell_idx}.gif",
                    save_all=True,
                    append_images=ims[1:],
                    duration=200,
                    loop=0)
    else:
        for i, im in enumerate(merged_im):
            # plt.imsave(dir / (f'{i}.jpeg'), np.transpose(im/255, (1, 2, 0)))
            plt.imsave(dir / (f'{i}.jpeg'), im / 255)


def save_cell_animation(
    hdf5_file: Path,
    save_path: Path,
    cell_idx: int,
    first_frame: int = 0,
    last_frame: int = 100,
    frame_size: int = 300,
    feature_names: Optional[list] = None,
    as_gif: bool = True,
    style: str = 'redline',
):
    """Animate a single cell alongside its feature time series.

    Left panel: cell image cropped and centred, outlined in yellow.
    Right panel: feature subplots.  style='redline' shows full traces with a
    red vertical line advancing each frame; style='progressive' draws feature
    values only up to the current frame.

    Args:
        hdf5_file: HDF5 dataset path.
        save_path: .gif path when as_gif=True, otherwise a directory of .png frames.
        cell_idx: Index of the cell to animate.
        first_frame: First frame (inclusive).
        last_frame: Last frame (exclusive).
        frame_size: Crop size in pixels around the cell centre.
        feature_names: Features to plot. Defaults to all 2-D features in the file.
        as_gif: Save as a single GIF when True, otherwise save per-frame PNGs.
        style: 'redline' or 'progressive'.
    """
    plt.rcParams["font.family"] = 'serif'

    n_frames = last_frame - first_frame
    phase_data = np.empty((n_frames, frame_size, frame_size))
    mask_data = np.empty((n_frames, frame_size, frame_size))

    with h5py.File(hdf5_file, 'r') as f:
        if feature_names is None:
            feature_names = [
                fn for fn in f['Cells']['Phase'].keys()
                if f['Cells']['Phase'][fn].ndim == 2
                and f['Cells']['Phase'][fn].shape[1] > cell_idx
            ]

        x_centres = tools.fill_nans(
            f['Cells']['Phase']['X'][first_frame:last_frame, cell_idx])
        y_centres = tools.fill_nans(
            f['Cells']['Phase']['Y'][first_frame:last_frame, cell_idx])

        image_size = f['Segmentations']['Phase'][0].shape

        feat_data = {
            fn: f['Cells']['Phase'][fn][first_frame:last_frame, cell_idx]
            for fn in feature_names
        }

        for idx, frame in tqdm(enumerate(range(first_frame, last_frame)),
                               total=n_frames,
                               desc='Loading frames'):
            ymin, ymax, xmin, xmax = mask_funcs.get_crop_indices(
                (y_centres[idx], x_centres[idx]), frame_size, image_size)
            phase_data[idx] = np.array(f['Images']['Phase'][frame])[xmin:xmax,
                                                                    ymin:ymax]
            mask_data[idx] = f['Segmentations']['Phase'][frame][xmin:xmax,
                                                                ymin:ymax]

    cell_mask = (mask_data == cell_idx)
    if not cell_mask.any():
        raise Exception(f'Cell of index {cell_idx} not found')

    cell_outline = mask_funcs.mask_outline(
        torch.tensor(cell_mask).byte().to(device), thickness=2).cpu().numpy()

    epi_data = np.zeros_like(phase_data)
    merged_im = tools.overlay_red_on_gray(phase_data,
                                          epi_data,
                                          normalize=False)
    merged_im[..., 0][cell_outline] = 255
    merged_im[..., 1][cell_outline] = 255

    if not as_gif:
        save_path.mkdir(parents=True, exist_ok=True)

    n_features = len(feature_names)
    frames_range = np.arange(first_frame, last_frame)
    gif_frames = []

    for idx in tqdm(range(n_frames), desc='Rendering frames'):
        current_frame = first_frame + idx

        fig = plt.figure(figsize=(12, max(0.7 * n_features, 4)))
        gs = gridspec.GridSpec(n_features,
                               2,
                               figure=fig,
                               width_ratios=[1, 2.5],
                               hspace=0.08,
                               wspace=0.35)

        ax_img = fig.add_subplot(gs[:, 0])
        ax_img.imshow(merged_im[idx].astype(np.uint8))
        ax_img.axis('off')
        ax_img.set_title(f'Frame {current_frame}', fontsize=9)

        for i, fn in enumerate(feature_names):
            ax = fig.add_subplot(gs[i, 1])
            vals = feat_data[fn]

            if style == 'progressive':
                ax.plot(frames_range[:idx + 1],
                        vals[:idx + 1],
                        color='k',
                        linewidth=1)
                ax.set_xlim(first_frame, last_frame - 1)
                y_min, y_max = np.nanmin(vals), np.nanmax(vals)
                if y_min != y_max:
                    ax.set_ylim(y_min, y_max)
            else:
                ax.plot(frames_range, vals, color='k', linewidth=1)
                ax.axvline(x=current_frame,
                           color='red',
                           linewidth=1.2,
                           alpha=0.8)
                ax.set_xlim(first_frame, last_frame - 1)

            wrapped = textwrap.fill(fn, width=16)
            ax.set_ylabel(wrapped,
                          rotation=0,
                          ha='right',
                          va='center',
                          labelpad=6,
                          fontsize=8)
            ax.tick_params(axis='both', labelsize=7)
            ax.grid(alpha=0.4)
            if i < n_features - 1:
                ax.set_xticklabels([])

        fig.axes[-1].set_xlabel('Frame', fontsize=9)
        # fig.suptitle(f'Cell {cell_idx}', fontsize=11, fontweight='bold')

        if as_gif:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            buf.seek(0)
            gif_frames.append(Image.open(buf).copy())
            buf.close()
        else:
            fig.savefig(save_path / f'{idx:04d}.png',
                        dpi=100,
                        bbox_inches='tight')

        plt.close(fig)

    if as_gif:
        out_path = save_path if str(save_path).endswith('.gif') else Path(
            str(save_path) + '.gif')
        gif_frames[0].save(out_path,
                           save_all=True,
                           append_images=gif_frames[1:],
                           duration=200,
                           loop=0)


# def save_cell_feature_plot(cell_idx: int, first_fame: int = 0, last_frame: int = 0) -> None:
#     plt.rcParams["font.family"] = 'serif'
#     with h5py.File(SETTINGS.DATASET, 'r') as f


def main():
    # save_tracked_images(Path('PhagoPred\\Datasets\\06_03\\H.h5'),
    #                     Path('temp/06_03_view'), 0, 100)
    # save_cell_images(Path('PhagoPred\\Datasets\\06_03\\H.h5'),
    #                  Path('temp/06_03_view_cell'),
    #                  0,
    #                  0,
    #                  100,
    #                  frame_size=300)
    feature_names = [
        'Area',
        # 'Circularity',
        'Perimeter',
        'Speed',
        # 'Displacement',
        # 'Mode 0',
        # 'Mode 1',
        # 'Mode 2',
        # 'Mode 3',
        # 'Mode 4',
        # 'Speed',
        # 'Alive Phagocytes within 100 pixels',
        # 'Alive Phagocytes within 250 pixels',
        'Alive Phagocytes within 500 pixels',
        # 'Dead Phagocytes within 100 pixels',
        # 'Dead Phagocytes within 250 pixels',
        # 'Dead Phagocytes within 500 pixels',

        # 'X',
        # 'Y',
        # 'CellDeath',
        'Total Fluorescence',
        'Fluorescence Distance Mean',
        # 'Fluorescence Distance Variance',
        # 'Inner Total Fluorescnece',
        # 'Outer Total FLuorescence',
        # 'External Fluorescence Intensity within 10 pixels',
        # 'External Fluorescence Intensity within 25 pixels',
        'External Fluorescence Intensity within 50 pixels',
        'Skeleton Length',
        'Skeleton Branch Points',
        # 'Skeleton End Points',
        # 'Skeleton Branch Length Mean',
        # 'Skeleton Branch Length Std',
        # 'Skeleton Branch Length Max',
    ]
    save_cell_animation(Path('PhagoPred\\Datasets\\06_03\\H.h5'),
                        Path('temp') / '06_03_0.gif',
                        0,
                        0,
                        100,
                        feature_names=feature_names)
    # save_cell_images(Path('temp'), 60, 0, 100, as_gif=True)
    # save_cell_images(Path(r'C:\Users\php23rjb\Downloads\temp') / 'test_track', cell_idx=347, first_frame=0, last_frame=50)
    # save_masks(Path('secondwithlight_masks'), 0, SETTINGS.NUM_FRAMES)


if __name__ == '__main__':
    main()
