from typing import Optional
import sys

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np

from PhagoPred import SETTINGS

def plot_cell_features(cell_idx: int, first_frame: int, last_frame: int, save_as: Path, feature_nmaes: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Plot time series of features for given cell index. If list of features is not given, plot all features."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()

        fig, axs = plt.subplots(len(feature_names), sharex=True, figsize=(10, 10))
        for i, feature_name in enumerate(feature_names):
            feature_values = f['Cells']['Phase'][feature_name][first_frame:last_frame, cell_idx]
            axs[i].plot(range(first_frame, last_frame), feature_values)
            axs[i].set_ylabel(feature_name, rotation=45, labelpad=20)
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame-1)

        fig.suptitle(f'Cell {cell_idx}')
        axs[-1].set(xlabel='Frame')

    plt.savefig(save_as)

    return fig

def plot_average_cell_features(first_frame: int, last_frame: int, save_as: Path, feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Plot time series of features averaged over all cells. If list of features is not given, plot all features."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()

        fig, axs = plt.subplots(len(feature_names), sharex=True, figsize=(10, 10))
        for i, feature_name in enumerate(feature_names):
            feature_values = f['Cells']['Phase'][feature_name][first_frame:last_frame]
            feature_means = np.nanmean(feature_values, axis=1)
            feature_stds = np.nanstd(feature_values, axis=1)

            axs[i].plot(range(first_frame, last_frame), feature_means)
            axs[i].fill_between(range(first_frame, last_frame),
                                feature_means-feature_stds,
                                feature_means+feature_stds,
                                alpha=0.5,
                                edgecolor=None)

            axs[i].set_ylabel(feature_name, rotation=45, labelpad=20)
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame-1)

        axs[-1].set(xlabel='Frame')

    plt.savefig(save_as)

    return fig

def plot_feature_correlations(save_as: Path, feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Show scatter plots and correlations between each feature, and histograms of each feature."""
    with h5py.File(SETTINGS.DATASET, 'r') as f:

        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()

        num_features = len(feature_names)
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(len(feature_names), len(feature_names), figsize=(num_features, num_features))
        cmap = plt.get_cmap('viridis_r')


        for i, feature_name_i in enumerate(feature_names):
            for j, feature_name_j in enumerate(feature_names):

                sys.stdout.write(f'\rPLotting Correlations: Progress {(i*num_features+j+1)/(num_features**2) * 100:.0f}%')
                sys.stdout.flush()

                feature_i = remove_outliers(f['Cells']['Phase'][feature_name_i][:].flatten())
                feature_j = remove_outliers(f['Cells']['Phase'][feature_name_j][:].flatten())

                R = corr_coefficient(feature_i, feature_j)
                color = cmap((R+1)/2)

                if i == 0:
                    axs[i, j].set_title(feature_name_j, rotation = 45)

                if j == 0:
                    axs[i, j].set_ylabel(feature_name_i, rotation = 45, labelpad=30)
                
                if j!=0 and i!=j:
                    axs[i, j].set_yticklabels([])
                
                if i != len(feature_names)-1:
                    axs[i, j].set_xticklabels([])

                # plot

                if i > j:
                    axs[i, j].scatter(feature_j, feature_i, s=0.1, linewidths=0, color=color)
                    # axs[i, j].grid()

                if i == j:
                    axs[i, j].hist(feature_i, bins=100, color='k')
                    # axs[i, j].grid()

                if i < j:
                    axs[i, j].text(0.5, 0.5, f'R = \n{R:.2f}', ha='center', va='center', transform=axs[i, j].transAxes,
                                    fontsize=14, fontweight='bold', color=color)
                    axs[i, j].axis('off')  
    plt.savefig(save_as)

    return fig

def remove_outliers(arr):
    """
    Using interquartile range (+/- 1.5*q)
    """
    no_nan = arr[~np.isnan(arr)]
    q1 = np.percentile(no_nan, 0.25)
    q3 = np.percentile(no_nan, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    arr[(arr < lower_bound)] = np.nan
    arr[(arr > upper_bound)] = np.nan
    return arr

# def corr_coefficient(array1, array2):
#     """"
#     Deals with np.nan when calculating correlation coefficeinet (numpy)
#     """
#     mask = ~np.isnan(array1) & ~np.isnan(array2)
#     return np.corrcoef(array1[mask], array2[mask])[0, 1]

def corr_coefficient(array1, array2):
    """
    Calculate correlation coefficient handling NaNs and constant arrays.
    """
    mask = ~np.isnan(array1) & ~np.isnan(array2)
    filtered_x = array1[mask]
    filtered_y = array2[mask]

    if len(filtered_x) < 2:
        return np.nan  # Not enough data to compute correlation

    if np.std(filtered_x) == 0 or np.std(filtered_y) == 0:
        return np.nan  # Constant arrays => undefined correlation

    return np.corrcoef(filtered_x, filtered_y)[0, 1]

def main():
    # plot_cell_features(50, 0, 50, Path('temp') / 'plot.png')
    # plot_average_cell_features(0, 50, Path('temp') / 'plot.png')
    plot_feature_correlations(Path('temp') / 'plot.png', feature_names=[
        'Area',
        'Circularity',
        'Perimeter',
        'Displacement',
        'Mode 0',
        'Mode 1',
        'Mode 2',
        'Mode 3',
        'Speed'
    ])

if __name__ == '__main__':
    main()