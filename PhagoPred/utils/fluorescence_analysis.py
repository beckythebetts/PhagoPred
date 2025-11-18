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
import scipy


from PhagoPred.utils import tools
from PhagoPred.utils.dataset_creation import to_8bit, compute_percentile_limits, replace_hot_pixels

results = []

from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import scipy.ndimage
import matplotlib.pyplot as plt

import napari
from PhagoPred.utils import tools, fluor_background_removal
from PhagoPred.utils.dataset_creation import to_8bit, compute_percentile_limits, replace_hot_pixels

plt.rcParams["font.family"] = 'serif'

class FluorescenceAnalysis:
    def __init__(self, strain_name: str, directory: Path, file_name: str, save_dir: Path = None):
        self.strain_name = strain_name
        self.directory = directory
        self.save_dir = save_dir or directory

        self.signal_tiff_file = directory / f'{file_name}_1' / f'{file_name}_1_MMStack_Pos0.ome.tif'

        self.phase_img, self.fluor_img = self.get_ims()

        self.masks = []
        self.combined_mask = None
        self.bg_mask = None

        # Save raw fluorescence image for reference
        plt.imsave(directory / f'{strain_name}.png', self.fluor_img, cmap='gray')
        try:
            self.combined_mask = plt.imread(self.save_dir / f'{self.strain_name}_combined_mask.png').astype(bool)[:, :, 0]
            print(self.combined_mask.shape)
            self.bg_mask = ~self.combined_mask
            # split combined_mask into disconenected regions
            # Label each connected region in the combined mask
            labeled_mask, num_labels = scipy.ndimage.label(self.combined_mask)

            # Create a list of boolean masks, one per connected region
            self.masks = [(labeled_mask == i) for i in range(1, num_labels + 1)]
        except: 
            self.get_masks()
        
        self.compute_metrics()
        self.plot_images_and_histogram()

    def get_ims(self) -> tuple[np.ndarray, np.ndarray]:
        img_stack = tifffile.imread(self.signal_tiff_file)
        self.phase_img = img_stack[2]
        self.fluor_img = img_stack[1].astype(np.int32)

        self.fluor_img = fluor_background_removal.replace_hot_pixels(self.fluor_img)
        self.fluor_img = fluor_background_removal.bg_removal(self.fluor_img)
        
        # crop_size = 100

        # def crop_from_center_corner(img, crop_size):
        #     h, w = img.shape[:2]
        #     # Top-left corner at the center of the image
        #     top = h // 2
        #     left = w // 2
        #     bottom = top + crop_size
        #     right = left + crop_size

        #     # Ensure we don’t go out of bounds
        #     bottom = min(bottom, h)
        #     right = min(right, w)

        #     return img[top:bottom, left:right]

        # self.phase_img = crop_from_center_corner(self.phase_img, crop_size)
        # self.fluor_img = crop_from_center_corner(self.fluor_img, crop_size)


        return self.phase_img, self.fluor_img

    def get_masks(self):
        """
        Open Napari for interactive ROI selection. Each ROI is stored as a boolean mask.
        """
        viewer = napari.Viewer()
        viewer.add_image(self.fluor_img, name='Fluorescence', colormap='red')
        viewer.add_image(self.phase_img, name='Phase', colormap='gray', blending='additive')

        shapes_layer = viewer.add_shapes(name='Masks', shape_type='polygon', edge_color='lime', face_color='lime', opacity=0.3)

        print("Draw polygons for bacteria, then close the Napari window when done.")
        napari.run()  # Blocks until the viewer window is closed

        masks = []
        for shape in shapes_layer.data:
            mask = np.zeros_like(self.fluor_img, dtype=bool)
            rr, cc = polygon_to_mask(shape[:, 1], shape[:, 0], self.fluor_img.shape)
            mask[rr, cc] = True
            masks.append(mask)

        self.masks = masks
        self.combined_mask = np.any(masks, axis=0) if masks else np.zeros_like(self.fluor_img, dtype=bool)
        self.bg_mask = ~self.combined_mask

        plt.imsave(self.save_dir / f'{self.strain_name}_combined_mask.png', self.combined_mask, cmap='gray')

    def compute_metrics(self) -> dict:
        print(len(self.masks))

        self.results = []
        columns = [
            'Mean',
            'Std',
            'Sum',
            'Pixels',
            'SNR',
            'SBR',
            'CV',
            '5perc',
            '95perc',
        ]
        indexes = []

        bg_pixels = self.fluor_img[self.bg_mask]
        pixels = np.sum(self.bg_mask)
        bg_mean = np.mean(bg_pixels)
        bg_std = np.std(bg_pixels)
        bg_5perc = np.percentile(bg_pixels, 5)
        bg_95perc = np.percentile(bg_pixels, 95)

        indexes.append('BG')

        result = {
                'Mean': bg_mean,
                'Std': bg_std,
                'Sum': np.nan,
                'Pixels': pixels,
                'SNR': np.nan,
                # 'CNR': np.nan,
                'SBR': np.nan,
                'CV': np.nan,
                '5perc': bg_5perc,
                '95perc': bg_95perc,
            }
        self.results.append(result)
        cell_idx = 0
        for mask in self.masks:
            signal_pixels = self.fluor_img[mask]
            signal_mean = np.mean(signal_pixels)
            signal_std = np.std(signal_pixels)
            signal_sum = np.sum(signal_pixels)
            signal_5perc = np.percentile(signal_pixels, 5)
            signal_95perc = np.percentile(signal_pixels, 95)
            pixels = np.sum(mask)
            if pixels == 0:
                continue
            snr = signal_mean / (bg_std + 1e-6)
            # cnr = (signal_mean - bg_mean) / (bg_std + 1e-6)
            cv = signal_std / (signal_mean + 1e-6)
            sbr = signal_mean / (bg_mean + 1e-6)

            result = {
                'Mean': signal_mean,
                'Std': signal_std,
                'Sum': signal_sum,
                'Pixels': pixels,
                'SNR': snr,
                'SBR': sbr,
                # 'CNR': cnr,
                'CV': cv,
                '5perc': signal_5perc,
                '95perc': signal_95perc,
            }
            self.results.append(result)
            indexes.append(f'Cell {cell_idx}')
            cell_idx += 1

        all_cell_results = np.array([list(cell_result.values()) for cell_result in self.results[1:]])
        means = np.mean(all_cell_results, axis=0)
        stds = np.std(all_cell_results, axis=0)
        all_5perc = np.percentile(all_cell_results, 5, axis=0)
        all_95_perc = np.percentile(all_cell_results, 95, axis=0)
        print(means, stds, all_5perc, all_95_perc)
        for vals in (means, stds, all_5perc, all_95_perc):
            self.results.append({metric: val for metric, val in zip(columns, vals)})
        indexes.append('Mean')
        indexes.append('Std')
        indexes.append('5perc')
        indexes.append('95perc')

        df = pd.DataFrame(self.results, columns=columns, index=indexes)
        df = df.round(3)
        df.to_csv(self.save_dir / f'{self.strain_name}_results.txt', index=True, sep='\t')
        return self.results

    def plot_images_and_histogram(self):
        
        
        if self.combined_mask is None:
            self.combined_mask = plt.imread(self.save_dir / f'{self.strain_name}_combined_mask.png').astype(bool)[:, :, 0]
            self.bg_mask = ~self.combined_mask
            
        fluor_signal = self.fluor_img[self.combined_mask]
        fluor_bg = self.fluor_img[self.bg_mask]
        
        fluor_signal = fluor_signal[fluor_signal > 0]
        fluor_bg = fluor_bg[fluor_bg > 0]
        
        crop = slice(300, 500), slice(300, 500)
        
        phase = self.phase_img[crop]
        fluor = self.fluor_img[crop]
        
        # Overlay
        phase_norm = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
        fluor_norm = (fluor - np.min(fluor)) / (np.max(fluor) - np.min(fluor))
        overlay = np.zeros((*phase.shape, 3))
        overlay[..., 0] = fluor_norm
        overlay[..., 1] = phase_norm
        overlay[..., 2] = phase_norm

        fig, axs = plt.subplots(1, 4, figsize=(11, 3), constrained_layout=True)
        axs[0].imshow(phase, cmap='gray'); axs[0].set_title('Phase Image'); axs[0].axis('off')
        axs[1].imshow(fluor, cmap='Reds'); axs[1].set_title('Fluorescence'); axs[1].axis('off')
        axs[2].imshow(overlay); axs[2].set_title('Overlay'); axs[2].axis('off')
        axs[3].hist(fluor_signal.ravel(), bins=20, color='red', alpha=0.7, density=True)
        axs[3].hist(fluor_bg.ravel(), bins=20, color='blue', alpha=0.5, density=True)
        axs[3].set_title('Fluorescence Intensity Histogram')
        axs[3].set_xlabel('Pixel Intensity')
        axs[3].set_ylabel('Normalised Frequency')
        axs[3].legend(['Signal', 'Background']); axs[3].set_title('Intensity Histogram')
        fig.suptitle(self.strain_name)
        plt.savefig(self.save_dir / f'{self.strain_name}_images.png')


# Helper function for converting polygon vertices to mask
def polygon_to_mask(x, y, shape):
    from skimage.draw import polygon
    rr, cc = polygon(y, x, shape)
    return rr, cc

def plot_results_line(directory: Union[Path, str], x_positions: list[float]) -> None:
    """
    Plots analysis results. All metrics except cell size are shown as line plots with shaded std.
    Cell size is plotted as grouped bars with legend.

    Args:
        directory: Directory with *_results.txt files.
        x_positions: Exposure times or other x-axis values for plotting.
    """
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Patch
    # import pandas as pd
    # import numpy as np

    plt.rcParams["font.family"] = 'serif'
    if isinstance(directory, str):
        directory = Path(directory)

    # Metrics to plot
    means_metrics = {
        'Signal Intensity': ('Mean', 'Mean'),
        'Cell Size / pixels': ('Mean', 'Pixels'),
        'Background Intensity': ('BG', 'Mean'),
        'SNR': ('Mean', 'SNR'),
        'SBR': ('Mean', 'SBR'),
        'CV': ('Mean', 'CV'),
    }

    std_metrics = {
        'Signal Intensity': ('Mean', 'Std'),
        'Background Intensity': ('BG', 'Std'),
        'SNR': ('Std', 'SNR'),
        'SBR': ('Std', 'SBR'),
        'CV': ('Std', 'CV'),
    }
    def get_cell_wise_spread(x: pd.DataFrame, metric:str) ->list[float, float]:
        return [x.at['5perc', metric], x.at['95perc', metric]]
    
    spread_metrics = {
        'Signal Intensity': lambda x: get_cell_wise_spread(x, 'Mean'),
        'Background Intensity': lambda x: [x.at['BG', '5perc'], x.at['BG', '95perc']],
        'Cell Size / pixels': lambda x: get_cell_wise_spread(x, 'Pixels'),
        'SNR': lambda x: get_cell_wise_spread(x, 'SNR'),
        'SBR': lambda x: get_cell_wise_spread(x, 'SBR'),
        'CV': lambda x: get_cell_wise_spread(x, 'CV'),
    }

    results_means = []
    results_stds = []
    results_spreads = []
    strains = []
    result_files = []

    # Load results
    for exposure in x_positions:
        file = directory / f'{exposure}s Exposure_results.txt'
        if not file.exists():
            print(f"Warning: File {file.name} not found. Skipping.")
            continue
        strain = file.stem.replace("_results", "")
        strains.append(strain)
        result_files.append(file)

        df = pd.read_csv(file, sep='\t', index_col=0)
        results_means.append({name: df.at[row, col] for name, (row, col) in means_metrics.items()})
        results_stds.append({name: df.at[row, col] for name, (row, col) in std_metrics.items()})
        results_spreads.append({name: func(df) for name, func in spread_metrics.items()})

    # Layout
    nrows = 2
    ncolumns = 3
    fig, axs = plt.subplots(nrows, ncolumns, sharex=False, figsize=(12, 8))
    axs = axs.flatten()

    # Colormap for strains (e.g., exposures)
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(len(strains))]

    line_metrics = [metric for metric in means_metrics.keys() if metric != 'Cell Size / pixels']
    # Plot line metrics
    for i, metric in enumerate(line_metrics):
        ax = axs[i]
        means = [result[metric] for result in results_means]
        stds = [result[metric] for result in results_stds]
        spreads = [result[metric] for result in results_spreads]

        means = np.array(means)
        stds = np.array(stds)
        spreads = np.transpose(np.array(spreads))
        spreads[0] = means - spreads[0]
        spreads[1] = spreads[1] - means
        print(means.shape, spreads.shape)
        ax.errorbar(x_positions, means, yerr=spreads, marker='o', color='k', capsize=5, label=metric)
        # ax.fill_between(x_positions, means - stds, means + stds, color='tab:red', alpha=0.5)
        ax.set_title(metric)
        ax.set_xlabel("Exposure Time (s)")
        ax.set_ylabel(metric)
        ax.grid()

    # Plot Cell Size as grouped bar plot
    cell_ax = axs[-1]
    width = 0.8
    x = np.arange(len(strains))

    size_metric = 'Cell Size / pixels'
    cell_means = []
    cell_means = [result[size_metric] for result in results_means]

    cell_spreads = [result[size_metric] for result in results_spreads]


    cell_spreads = np.transpose(np.array(cell_spreads))

    
    cell_spreads[0] = cell_means - cell_spreads[0]
    cell_spreads[1] = cell_spreads[1] - cell_means
    
    bars = cell_ax.bar(x, cell_means, yerr=cell_spreads, capsize=5, color=colors)
    cell_ax.set_xticks(x)
    cell_ax.set_xticklabels(strains, rotation=45)
    cell_ax.set_title("Cell Size / pixels")
    cell_ax.set_ylabel("Pixels")
    
    plt.tight_layout()
    plt.savefig(directory / 'results_lineplot.png')


    
def combine_images(directory: Union[Path, str], exposures: list) -> None:
    if isinstance(directory, str):
        directory = Path(directory)
    stacked_image = []
    # for file in directory.glob("*_images.png"):
    # for file in (directory / '1s Exposure_images.png', directory / '2.5s Exposure_images.png', directory / '5s Exposure_images.png', directory / '10s Exposure_images.png'):
    for exposure in exposures:
        file = directory / f'{exposure}s Exposure_images.png'
        stacked_image.append(plt.imread(file))
    stacked_image = np.concatenate(stacked_image, axis=0)
    plt.imsave(directory / 'image_all.png', stacked_image)



def main():
    # FluorescenceAnalysis("1s Exposure", Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / 'exposure_test', "mcherry_exposure_1s")
    # directory = Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / 'ExposureTest1411' / '1ml'
    exposures = ['0.5', '1.0', '2.5', '5.0', '10.0']
    # for exposure in ('0.5', '1.0', '2.5', '5.0', '10.0'):
    #     FluorescenceAnalysis(f"{exposure}s Exposure", directory, f"{exposure.replace('.', '_')}s")
    directory =  Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / '10_11_exposures'
    exposures = ['1', '2.5', '5', '10']
    for exposure in exposures:
        FluorescenceAnalysis(f"{exposure}s Exposure", directory, f"{exposure.replace('.', '_')}s")

    # exposures = ['1.0', '2.5', '5.0', '10.0']
    plot_results_line(directory, exposures)

    combine_images(directory, exposures)

if __name__ == "__main__":
    main()
    