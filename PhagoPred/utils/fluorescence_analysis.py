from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as PlotPath
from matplotlib.patches import Patch
import pandas as pd
from pathlib import Path
import tifffile

from PhagoPred.utils import tools

results = []

class FluorescenceAnalysis():
    def __init__(self, strain_name: str, tiff_file: Path, save_dir: Path = None) -> None:
        self.strain_name = strain_name

        self.tiff_file = tiff_file
        if isinstance(self.tiff_file, str):
            self.tiff_file = Path(self.tiff_file)

        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = self.tiff_file.parent.parent

        self.crop = (slice(1200, 1300), slice(1200, 1300))

        self.phase_img, self.fluor_img = self.get_ims()

        self.masks = []

        # self.get_masks()
        # self.compute_metrics()
        self.plot_images_and_histogram()

    def get_ims(self) -> tuple[np.ndarray, np.ndarray]:
        img_stack = tifffile.imread(self.tiff_file)

        if img_stack.shape[0] != 2:
            raise ValueError("Expected 2-channel TIFF (shape[0] should be 2).")

        self.phase_img = img_stack[0][self.crop]
        self.fluor_img = img_stack[1][self.crop].astype(np.float32)
        return self.phase_img, self.fluor_img

    def get_masks(self):
        fig, ax = plt.subplots()
        ax.imshow(self.phase_img, cmap='gray')
        plt.title("Draw masks around bacteria. Press Enter when done.")

        self.masks = []
        selector = [None]  # Mutable wrapper for PolygonSelector

        def onselect(verts):
            path = PlotPath(verts)
            y, x = np.meshgrid(
                np.arange(self.phase_img.shape[0]),
                np.arange(self.phase_img.shape[1]),
                indexing='ij'
            )
            points = np.vstack((x.flatten(), y.flatten())).T
            mask = path.contains_points(points).reshape(self.phase_img.shape)
            self.masks.append(mask)

            # # Show visual feedback
            # ax.imshow(mask, alpha=0.3, cmap='Reds')

            # Recreate selector to allow another polygon
            fig.canvas.mpl_disconnect(selector[0]._key_press_cid)
            start_selector()

        def start_selector():
            selector[0] = PolygonSelector(ax, onselect, useblit=True)
            selector[0]._key_press_cid = fig.canvas.mpl_connect('key_press_event', on_key)

        def on_key(event):
            if event.key == 'enter':
                plt.close(fig)

        # Start the first selector
        start_selector()

        plt.show()

        # Combine masks after all polygons are drawn
        if self.masks:
            self.combined_mask = np.any(self.masks, axis=0)
        else:
            self.combined_mask = np.zeros_like(self.phase_img, dtype=bool)

        self.bg_mask = ~self.combined_mask

    def compute_metrics(self) -> dict:
        print(len(self.masks))

        self.results = []
        columns = [
            'Mean',
            'Std',
            'Sum',
            'Pixels',
            'SNR',
            'CNR',
            'CV',
        ]
        indexes = []

        bg_pixels = self.fluor_img[self.bg_mask]
        pixels = np.sum(self.bg_mask)
        bg_mean = np.mean(bg_pixels)
        bg_std = np.std(bg_pixels)

        indexes.append('BG')

        result = {
                'Mean': bg_mean,
                'Std': bg_std,
                'Sum': np.nan,
                'Pixels': pixels,
                'SNR': np.nan,
                'CNR': np.nan,
                'CV': np.nan,
            }
        self.results.append(result)
        cell_idx = 0
        for mask in self.masks:
            signal_pixels = self.fluor_img[mask]
            signal_mean = np.mean(signal_pixels)
            signal_std = np.std(signal_pixels)
            signal_sum = np.sum(signal_pixels)
            pixels = np.sum(mask)
            if pixels == 0:
                continue
            snr = signal_mean / (bg_std + 1e-6)
            cnr = (signal_mean - bg_mean) / (bg_std + 1e-6)
            cv = signal_std / (signal_mean + 1e-6)

            result = {
                'Mean': signal_mean,
                'Std': signal_std,
                'Sum': signal_sum,
                'Pixels': pixels,
                'SNR': snr,
                'CNR': cnr,
                'CV': cv
            }
            self.results.append(result)
            indexes.append(f'Cell {cell_idx}')
            cell_idx += 1

        all_cell_results = np.array([list(cell_result.values()) for cell_result in self.results[1:]])
        means = np.mean(all_cell_results, axis=0)
        stds = np.std(all_cell_results, axis=0)
        for vals in (means, stds):
            self.results.append({metric: val for metric, val in zip(columns, vals)})
        indexes.append('Mean')
        indexes.append('Std')

        df = pd.DataFrame(self.results, columns=columns, index=indexes)
        df = df.round(3)
        df.to_csv(self.save_dir / f'{self.strain_name}_results.txt', index=True, sep='\t')
        print(f"\nSaved {len(self.results)} entries to {self.save_dir / 'results.txt'}")
        return self.results
    
    def plot_images_and_histogram(self):
        plt.rcParams["font.family"] = 'serif'

        phase = self.phase_img
        fluor = self.fluor_img
        fluor = tools.threshold_image(fluor, lower_percentile=0, upper_percentile=99.9)

        # Normalize for overlay
        phase_norm = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
        fluor_norm = (fluor - np.min(fluor)) / (np.max(fluor) - np.min(fluor))

        # Create RGB overlay: Grayscale phase + red fluorescence
        overlay = np.zeros((*phase.shape, 3))
        overlay[..., 0] = fluor_norm      # Red channel
        overlay[..., 1] = phase_norm      # Green channel (optional)
        overlay[..., 2] = phase_norm      # Blue channel (optional)

        # Plot images side by side + overlay + histogram
        fig, axs = plt.subplots(1, 4, figsize=(18, 5))

        axs[0].imshow(phase, cmap='gray')
        axs[0].set_title('Phase Image')
        axs[0].axis('off')

        axs[1].imshow(fluor, cmap='Reds')
        axs[1].set_title('Fluorescence (Red)')
        axs[1].axis('off')

        axs[2].imshow(overlay)
        axs[2].set_title('Overlay')
        axs[2].axis('off')

        axs[3].hist(fluor.ravel(), bins=50, color='red', alpha=0.7)
        axs[3].set_title('Fluorescence Intensity Histogram')
        axs[3].set_xlabel('Pixel Intensity')
        axs[3].set_ylabel('Frequency')

        plt.suptitle(self.strain_name)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{self.strain_name}_images.png')
    # def save_ims(self):

    #     plt.imsave(self.save_dir / 'f{self.strain_name}_images.png')
    

def plot_results(directory: Union[Path, str]) -> None:
    """"""
    plt.rcParams["font.family"] = 'serif'
    if isinstance(directory, str):
        directory = Path(directory)

    nrows = 2
    ncolumns = 3

    data_means = {
        'Signal Intensity': lambda x: x.at['Mean', 'Mean'],
        'Background Intensity': lambda x: x.at['BG', 'Mean'],
        'Cell Size / pixels': lambda x: x.at['Mean', 'Pixels'],
        'SNR': lambda x: x.at['Mean', 'SNR'],
        'CNR': lambda x: x.at['Mean', 'CNR'],
        'CV': lambda x: x.at['Mean', 'CV'],
    }
    data_stds = {
        'Signal Intensity': lambda x: x.at['Mean', 'Std'],
        'Background Intensity': lambda x: x.at['BG', 'Std'],
        'Cell Size / pixels': lambda x: x.at['Std', 'Pixels'],
        'SNR': lambda x: x.at['Std', 'SNR'],
        'CNR': lambda x: x.at['Std', 'CNR'],
        'CV': lambda x: x.at['Std', 'CV'],
    }

    results_means = []
    results_stds = []
    strains = []

    # Load all result files
    for file in directory.glob("*_results.txt"):
        strain = file.stem.replace("_results", "")
        strains.append(strain)

        df = pd.read_csv(file, sep='\t', index_col=0)
        results_means.append({name: func(df) for name, func in data_means.items()})
        results_stds.append({name: func(df) for name, func in data_stds.items()})

    fig, axs = plt.subplots(nrows, ncolumns, sharex=True, figsize=(10, 5))
    axs = axs.flatten()
    colours = [plt.get_cmap('tab10').colors[i] for i in range(len(strains))]

    for ax, metric in zip(axs, data_means.keys()):
        metric_means = [strain_means[metric] for strain_means in results_means]
        metric_stds = [strain_stds[metric] for strain_stds in results_stds]
        ax.bar(range(len(strains)), metric_means, yerr=metric_stds, color=colours, capsize=5)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        # ax.set_title(metric)
        # ax.set_xticklabels(strains, rotation=45, ha='right') 
    # After defining `strains` and `colours`
    legend_handles = [Patch(color=color, label=strain) for strain, color in zip(strains, colours)]
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    fig.legend(
        handles=legend_handles,
        loc='upper right',
        frameon=False,
    )
    plt.savefig(directory / 'results.png')

def combine_images(directory: Union[Path, str]) -> None:
    if isinstance(directory, str):
        directory = Path(directory)
    stacked_image = []
    for file in directory.glob("*_images.png"):
        stacked_image.append(plt.imread(file))
    stacked_image = np.concatenate(stacked_image, axis=0)
    plt.imsave(directory / 'image_all.png', stacked_image)



def main():
    FluorescenceAnalysis("SH1000 - mCherry",
                         "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\4631_SH1000_mC_1\\4631_SH1000_mC_1_MMStack_Pos0.ome.tif",
                         )
    FluorescenceAnalysis("JE2 - mCherry",
                         "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\4634_JE2_mC_1\\4634_JE2_mC_1_MMStack_Pos0.ome.tif")
    FluorescenceAnalysis("SH1000 - smURFP",
                         "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\5522_SH1000_Sm_1\\5522_SH1000_Sm_1_MMStack_Pos0.ome.tif")
    FluorescenceAnalysis("JE2 - smURFP",
                         "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\5525_JE2_Sm_1\\5525_JE2_Sm_1_MMStack_Pos0.ome.tif")
    combine_images("C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo")
    # plot_results("C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo")

if __name__ == "__main__":
    main()
