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

class FluorescenceAnalysis():
    def __init__(self, strain_name: str, directory: Path, file_name: str, save_dir: Path = None) -> None:
        print(directory)
        self.strain_name = strain_name
        self.combined_mask = None
        self.bg_mask = None

        # self.tiff_file = tiff_file

        self.signal_tiff_file = directory / f'{file_name}_1' / f'{file_name}_1_MMStack_Pos0.ome.tif'
        # self.background_tiff_file = directory / f'{file_name}_background_1' / f'{file_name}_background_1_MMStack_Pos0.ome.tif'

        self.save_dir = save_dir
        if self.save_dir is None:
            self.save_dir = directory

        # self.crop = (slice(1200, 1300), slice(1200, 1300))

        self.phase_img, self.fluor_img = self.get_ims()

        self.masks = []

        plt.imsave(directory / f'{strain_name}.png', self.fluor_img)
        # try:
        #     self.combined_mask = plt.imread(self.save_dir / f'{self.strain_name}_combined_mask.png').astype(bool)[:, :, 0]
        #     print(self.combined_mask.shape)
        #     self.bg_mask = ~self.combined_mask
        #     # split combined_mask into disconenected regions
        #     # Label each connected region in the combined mask
        #     labeled_mask, num_labels = scipy.ndimage.label(self.combined_mask)

        #     # Create a list of boolean masks, one per connected region
        #     self.masks = [(labeled_mask == i) for i in range(1, num_labels + 1)]
        # except: 
        #     self.get_masks()
            
        # self.compute_metrics()
        # self.plot_images_and_histogram()

    def get_ims(self) -> tuple[np.ndarray, np.ndarray]:
        img_stack = tifffile.imread(self.signal_tiff_file)
        # background_stack = tifffile.imread(self.background_tiff_file)

        # if img_stack.shape[0] != 2:
            # raise ValueError("Expected 2-channel TIFF (shape[0] should be 2).")

        self.phase_img = img_stack[2]
        self.fluor_img = img_stack[1].astype(np.int32)
            # - background_stack.astype(np.int32)
        # self.fluor_img = background_stack.astype(np.int16)
        print(self.fluor_img.shape)
        
        # determine 'hot pixels'
        # min_val, max_val = compute_percentile_limits(self.fluor_img, 99.9, 100)
        self.fluor_img = replace_hot_pixels(self.fluor_img, 99.9, 3)
        # self.fluor_img = scipy.ndimage.median_filter(self.fluor_img, size=10)
        # self.fluor_img = to_8bit(self.fluor_img, min_val, max_val)
        # self.fluor_img = scipy.ndimage.gaussian_filter(self.fluor_img, sigma=3)
        # Gaussian filtering for vignette rmeoval?
        bg_estimate = scipy.ndimage.gaussian_filter(self.fluor_img, sigma=50)
        self.fluor_img = self.fluor_img - bg_estimate
        self.fluor_img = np.clip(self.fluor_img, 0, None)
        self.fluor_img = scipy.ndimage.median_filter(self.fluor_img, size=10)
        self.fluor_img = scipy.ndimage.gaussian_filter(self.fluor_img, sigma=3)
        return self.phase_img, self.fluor_img

    def get_masks(self):
        fig, ax = plt.subplots()
        ax.imshow(self.fluor_img, cmap='gray')
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
        print(f"\nSaved {len(self.results)} entries to {self.save_dir / 'results.txt'}")
        return self.results
    
    def plot_images_and_histogram(self):
        plt.rcParams["font.family"] = 'serif'

        
        crop = slice(150, 350), slice(300, 500)
        crop = slice(None), slice(None)
        phase = self.phase_img[crop]
        fluor = self.fluor_img[crop]
        fluor = tools.threshold_image(fluor, lower_percentile=0, upper_percentile=99.9)
        
        if self.combined_mask is None:
            self.combined_mask = plt.imread(self.save_dir / f'{self.strain_name}_combined_mask.png').astype(bool)[:, :, 0]
            print(self.combined_mask.shape)
            self.bg_mask = ~self.combined_mask
        fluor_signal = fluor[self.combined_mask[crop]]
        fluor_bg = fluor[self.bg_mask[crop]]

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

        fluor_signal = fluor_signal[fluor_signal>0]
        fluor_bg = fluor_bg[fluor_bg>0]
        axs[3].hist(fluor_signal.ravel(), bins=50, color='red', alpha=0.7, density=True)
        axs[3].hist(fluor_bg.ravel(), bins=50, color='blue', alpha=0.5, density=True)
        axs[3].legend(['Signal', 'Background'])
        axs[3].set_title('Fluorescence Intensity Histogram')
        axs[3].set_xlabel('Pixel Intensity')
        axs[3].set_ylabel('Normalised Frequency')
        # axs[3].set_yscale('log')

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
        # 'CNR': lambda x: x.at['Mean', 'CNR'],
        'SBR': lambda x: x.at['Mean', 'SBR'],
        'CV': lambda x: x.at['Mean', 'CV'],
    }
    data_stds = {
        'Signal Intensity': lambda x: x.at['Mean', 'Std'],
        'Background Intensity': lambda x: x.at['BG', 'Std'],
        'Cell Size / pixels': lambda x: x.at['Std', 'Pixels'],
        'SNR': lambda x: x.at['Std', 'SNR'],
        # 'CNR': lambda x: x.at['Std', 'CNR'],
        'SBR': lambda x: x.at['Std', 'SBR'],
        'CV': lambda x: x.at['Std', 'CV'],
    }

    results_means = []
    results_stds = []
    strains = []

    # Load all result files
    # for file in directory.glob("*_results.txt"):
    for file in (directory / '1s Exposure_results.txt', directory / '2.5s Exposure_results.txt', directory / '5s Exposure_results.txt', directory / '10s Exposure_results.txt'):
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
    for exposure in ('1', '2.5', '5', '10'):
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
    cmap = plt.get_cmap('tab10')
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
        print(spreads)
        spreads[0] = means - spreads[0]
        spreads[1] = spreads[1] - means
        print(means)
        print(spreads)
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
    cell_stds = []
    cell_means = [result[size_metric] for result in results_means]
    # cell_stds = [result[size_metric] for result in results_stds]
    cell_spreads = [result[size_metric] for result in results_spreads]
    # for file in result_files:
    #     df = pd.read_csv(file, sep='\t', index_col=0)
    #     cell_means.append(df.at[cell_size_metric[0], cell_size_metric[1]])
    #     cell_stds.append(df.at[cell_size_std[0], cell_size_std[1]])

    cell_spreads = np.transpose(np.array(cell_spreads))
    print(cell_means)
    print(cell_spreads)
    
    cell_spreads[0] = cell_means - cell_spreads[0]
    cell_spreads[1] = cell_spreads[1] - cell_means
    print(cell_spreads)
    
    bars = cell_ax.bar(x, cell_means, yerr=cell_spreads, capsize=5, color=colors)
    cell_ax.set_xticks(x)
    cell_ax.set_xticklabels(strains, rotation=45)
    cell_ax.set_title("Cell Size / pixels")
    cell_ax.set_ylabel("Pixels")

    # Legend for cell size bars
    # legend_handles = [Patch(color=color, label=strain) for strain, color in zip(strains, colors)]
    # fig.legend(handles=legend_handles, loc='upper right', frameon=False)
    # cell_ax.legend(handles=legend_handles)

    # plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.tight_layout()
    plt.savefig(directory / 'results_lineplot.png')
    # plt.show()

    
def combine_images(directory: Union[Path, str]) -> None:
    if isinstance(directory, str):
        directory = Path(directory)
    stacked_image = []
    # for file in directory.glob("*_images.png"):
    for file in (directory / '1s Exposure_images.png', directory / '2.5s Exposure_images.png', directory / '5s Exposure_images.png', directory / '10s Exposure_images.png'):
        stacked_image.append(plt.imread(file))
    stacked_image = np.concatenate(stacked_image, axis=0)
    plt.imsave(directory / 'image_all.png', stacked_image)



def main():
    # FluorescenceAnalysis("1s Exposure", Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / 'exposure_test', "mcherry_exposure_1s")
    for exposure in ('1', '2.5', '5', '10', '20'):
        FluorescenceAnalysis(f"{exposure}s Exposure", Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / '11_11_pulsed_exposures', f"{exposure.replace('.', '_')}s")

    # FluorescenceAnalysis("2.5s E")
    # FluorescenceAnalysis("SH1000 - mCherry",
    #                      "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\4631_SH1000_mC_1\\4631_SH1000_mC_1_MMStack_Pos0.ome.tif",
    #                      )
    # FluorescenceAnalysis("JE2 - mCherry",
    #                      "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\4634_JE2_mC_1\\4634_JE2_mC_1_MMStack_Pos0.ome.tif")
    # FluorescenceAnalysis("SH1000 - smURFP",
    #                      "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\5522_SH1000_Sm_1\\5522_SH1000_Sm_1_MMStack_Pos0.ome.tif")
    # FluorescenceAnalysis("JE2 - smURFP",
    #                      "C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo\\5525_JE2_Sm_1\\5525_JE2_Sm_1_MMStack_Pos0.ome.tif")
    # combine_images("C:\\Users\\php23rjb\\Downloads\\14_08_stap_redo")
    # plot_results("C:\\Users\\php23rjb\\Downloads\\temp\\exposure_test")
    plot_results_line("C:\\Users\\php23rjb\\Downloads\\temp\\exposure_test", [1, 2.5, 5, 10])

    # combine_images(Path("C:\\Users") / 'php23rjb' / 'Downloads' / 'temp' / 'exposure_test')

if __name__ == "__main__":
    main()
    