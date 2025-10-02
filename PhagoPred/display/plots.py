from typing import Optional
import sys

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.extract_features import CellType

plt.rcParams["font.family"] = 'serif'

def plot_cell_features(cell_idx: int, first_frame: int, last_frame: int, save_as: Path, feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
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

def plot_average_cell_features(save_as: Path, first_frame: Optional[int] = None, last_frame: Optional[int] = None, feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Plot time series of features averaged over all cells. If list of features is not given, plot all features."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()
        if first_frame is None:
            first_frame = 0
        if last_frame is None:
            last_frame = f['Cells']['Phase'][feature_names[0]].shape[0]

        fig, axs = plt.subplots(len(feature_names), sharex=True, figsize=(10, 10))
        for i, feature_name in enumerate(tqdm(feature_names)):
            feature_values = f['Cells']['Phase'][feature_name][first_frame:last_frame]
            # feature_means = np.nanmean(feature_values, axis=1)
            # feature_stds = np.nanstd(feature_values, axis=1)

            # Compute nanmean and nanstd safely
            feature_means = np.full(feature_values.shape[0], np.nan)
            feature_stds = np.full(feature_values.shape[0], np.nan)

            for t in range(feature_values.shape[0]):
                if np.isnan(feature_values[t]).all():
                    continue  # Leave mean and std as NaN for that timepoint
                feature_means[t] = np.nanmean(feature_values[t])
                feature_stds[t] = np.nanstd(feature_values[t])


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


        for i, feature_name_i in enumerate(tqdm(feature_names)):
            for j, feature_name_j in enumerate(feature_names):

                # sys.stdout.write(f'\rPLotting Correlations: Progress {(i*num_features+j+1)/(num_features**2) * 100:.0f}%')
                # sys.stdout.flush()

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

    return fig#

def plot_num_dead_cells(save_as: Optional[Path]=None, first_frame: Optional[int] = None, last_frame: Optional[int] = None) -> 'matplotlib.figure.Figure':
    """Plot number of dead cells over time."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        if first_frame is None:
            first_frame = 0
        if last_frame is None:
            last_frame = f['Cells']['Phase']['CellDeath'].shape[0]

        # num_dead_cells = np.sum(f['Cells']['Phase']['CellDeath'][first_frame:last_frame], axis=1)
        num_alive_cells = np.nansum(f['Cells']['Phase']['Macrophage'][first_frame:last_frame], axis=1)
        num_dead_cells = np.nansum(f['Cells']['Phase']['Dead Macrophage'][first_frame:last_frame], axis=1)
        total_cells = num_alive_cells + num_dead_cells

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(first_frame, last_frame), num_dead_cells, label='Dead Cells', color='red')
        ax.plot(range(first_frame, last_frame), num_alive_cells, label='Alive Cells', color='blue')
        ax.plot(range(first_frame, last_frame), total_cells, label='Total Cells', color='black', linestyle='--')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Number of Cells')
        ax.legend()
        ax.grid()
    if save_as is not None:
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

def plot_death_frames_hist(death_frames_txt: Path, save_as: Path):
    death_frames = pd.read_csv(death_frames_txt, sep="|", skiprows=2, engine='python')
    death_frames = death_frames.iloc[:, 1:-1]
    death_frames.columns = ['Cell Idx', 'True', 'Predicted']
    death_frames = death_frames[['Cell Idx', 'True', 'Predicted']].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    valid_rows = death_frames[death_frames['True'].str.isnumeric() & death_frames['Predicted'].str.isnumeric()].copy()

    valid_rows['True'] = valid_rows['True'].astype(int)
    valid_rows['Predicted'] = valid_rows['Predicted'].astype(int)

    errors = valid_rows['Predicted'] - valid_rows['True']
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Number of Cells')
    plt.title('Histogram of Cell Death Frame Prediction Errors')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_as)

def plot_two_death_frame_hists(
    death_frames_txt1: Path,
    death_frames_txt2: Path,
    label1: str,
    label2: str,
    save_as: Path
):
    def load_and_compute_errors(path: Path):
        df = pd.read_csv(path, sep="|", skiprows=2, engine='python')
        df = df.iloc[:, 1:-1]
        df.columns = ['Cell Idx', 'True', 'Predicted']
        df = df[['Cell Idx', 'True', 'Predicted']].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        valid = df[df['True'].str.isnumeric() & df['Predicted'].str.isnumeric()].copy()
        valid['True'] = valid['True'].astype(int)
        valid['Predicted'] = valid['Predicted'].astype(int)
        errors = valid['Predicted'] - valid['True']
        return errors

    errors1 = load_and_compute_errors(death_frames_txt1)
    errors2 = load_and_compute_errors(death_frames_txt2)
    all_errors = pd.concat([errors1, errors2])
    min_edge = all_errors.min()
    max_edge = all_errors.max()
    bins = 30  # or any number you choose
    bin_edges = np.linspace(min_edge, max_edge, bins + 1)

    plt.figure(figsize=(8, 6))
    plt.hist(errors1, bins=bin_edges, alpha=0.6, color='blue', edgecolor='black', label=label1)
    plt.hist(errors2, bins=bin_edges, alpha=0.6, color='red', edgecolor='black', label=label2)

    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Number of Cells')
    plt.title('Comparison of Cell Death Frame Prediction Errors')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

def plot_death_frame(death_frames_txt: Path, save_as: Path):
    death_frames = pd.read_csv(death_frames_txt, sep="|", skiprows=2, engine='python')
    death_frames = death_frames.iloc[:, 1:-1]
    death_frames.columns = ['Cell Idx', 'True', 'Predicted']
    death_frames = death_frames[['Cell Idx', 'True', 'Predicted']].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    valid_rows = death_frames[death_frames['True'].str.isnumeric() & death_frames['Predicted'].str.isnumeric()].copy()

    valid_rows['True'] = valid_rows['True'].astype(int)
    valid_rows['Predicted'] = valid_rows['Predicted'].astype(int)

    plt.figure(figsize=(8, 6))
    plt.scatter(valid_rows['True'], valid_rows['Predicted'], color='blue', label='Cell Death Prediction', marker='.')
    plt.plot([valid_rows['True'].min(), valid_rows['True'].max()],
            [valid_rows['True'].min(), valid_rows['True'].max()],
            color='red', linestyle='--', label='Perfect Prediction')

    plt.xlabel('True Death Frame')
    plt.ylabel('Predicted Death Frame')
    plt.title('Predicted vs. True Cell Death Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as)
    avg_error = np.sum(np.abs(valid_rows['True'] - valid_rows['Predicted'])) / len(valid_rows['True'])
    print(f"\nAverage error in cell death estimation is {avg_error} frames.\n")

def km_plot(hdf5_files: list[Path], labels: list[str], save_as: Path, time_steps: tuple[int] = (1, 1)):
    """Plot the Kaplan Meier estimator for hdf5_file/s."""
    
    if isinstance(hdf5_files, Path):
        hdf5_files = [hdf5_files]
        
    cmap = plt.get_cmap('Set1')
    colours = [cmap(i) for i in range(len(hdf5_files))]
    def survival_estimator(d_is: np.ndarray, n_is:np.ndarray):
        """Estimate surival funciotn at each time t_i.
        args:
            d_is = number of events(death) at each time t_i
            n_is = number of samples still alive at each time t_i.
        returns:
            S [float]
        """
        return np.cumprod(1 - (d_is)/(n_is))
    
    for hdf5_file, label, colour, time_step in zip(hdf5_files, labels, colours, time_steps):
        
        with h5py.File(hdf5_file, 'r') as f:
            cell_deaths_arr = f['Cells']['Phase']['CellDeath']
            num_frames = cell_deaths_arr.shape[0]
            
            print(num_frames)
            cell_deaths = cell_deaths_arr[0]
            
            # Load only 'Area' to compute start/end frames
            area_data = CellType('Phase').get_features_xr(f, features=['Area'])['Area'].transpose('Cell Index', 'Frame').values  # shape: (num_cells, num_frames)

            # Use numpy vectorization for speed
            not_nan_mask = ~np.isnan(area_data)
            # First valid frame (start frame)
            # Last valid frame (end frame)
            reversed_mask = not_nan_mask[:, ::-1]
            last_frames = area_data.shape[1] - 1 - reversed_mask.argmax(axis=1)
        ds, ns = np.zeros(shape=num_frames), np.zeros(shape=num_frames)
        for cell_death, last_frame in zip(cell_deaths, last_frames):
            if np.isnan(cell_death):
                ns[np.arange(0, last_frame).astype(int)] += 1
            else:
                ns[np.arange(0, cell_death).astype(int)] += 1
                ds[cell_death.astype(int)] += 1
        mask = ds > 0
        ds = ds[mask]
        ns = ns[mask]
        ts = np.nonzero(mask)[0]
        ts = ts * time_step
        print(ts)
        print(mask.shape, ds.shape, ns.shape, ts.shape)
        plt.step(ts, survival_estimator(ds, ns), where='post', label=label, color=colour)
    
    plt.xlabel('Time / minutes')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid()
    plt.savefig(save_as)
    
def compare_cell_features(hdf5_files: list[Path], labels: list[str], feature: str, save_as: Path) -> None:
    """Compare the histograms of a given cell feature between two datasets.
    If compare_dead = true, plot separate histograms for cells which die during observation vs those that dont."""
    
    if isinstance(hdf5_files, Path):
        hdf5_files = [hdf5_files]
        
    cmap = plt.get_cmap('Set1')
    num_colours = len(hdf5_files)
    colours = [cmap(i) for i in range(len(hdf5_files))]
        
    for hdf5_file, label, colour in zip(hdf5_files, labels, colours):
        with h5py.File(hdf5_file, 'r') as f:
            features_ds = f['Cells']['Phase'][feature][:]
            cell_deaths_arr = f['Cells']['Phase']['CellDeath']
            num_frames = cell_deaths_arr.shape[0]
            
            print(num_frames)
            cell_deaths = cell_deaths_arr[0]
            for i, cell_death in enumerate(cell_deaths):
                if not np.isnan(cell_death):
                    features_ds[int(cell_death):, i] = np.nan
            
            
            plt.hist(features_ds.ravel(), bins=50, alpha=0.5, color=colour, label=label, density=True)
    
    plt.xlabel(feature)
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()
    plt.savefig(save_as)

def plot_alive_vs_dead_feature(hdf5_file: Path, feature: str, save_as: Path) -> None:
    """Plot histograms of a given feature for cells that survive vs cells that die."""
    
    with h5py.File(hdf5_file, 'r') as f:
        features_ds = f['Cells']['Phase'][feature][:]          # shape: (num_frames, num_cells)
        cell_deaths = f['Cells']['Phase']['CellDeath'][0]      # shape: (num_cells,)
        
        num_frames, num_cells = features_ds.shape

        alive_values = []
        dead_values = []

        for cell_idx in range(num_cells):
            death_frame = cell_deaths[cell_idx]

            if np.isnan(death_frame):
                # Alive — include all frames
                alive_values.append(features_ds[:, cell_idx])
            else:
                # Dead — include up to death frame (exclude death frame itself)
                dead_values.append(features_ds[:int(death_frame), cell_idx])

        # Flatten all values
        alive_values_flat = np.concatenate(alive_values) if alive_values else np.array([])
        dead_values_flat = np.concatenate(dead_values) if dead_values else np.array([])
        
        alive_values_flat = alive_values_flat[~np.isnan(alive_values_flat)]
        dead_values_flat = dead_values_flat[~np.isnan(dead_values_flat)]
        alive_values_flat = alive_values_flat[alive_values_flat <= np.percentile(alive_values_flat, 99)]
        dead_values_flat = dead_values_flat[dead_values_flat <= np.percentile(dead_values_flat, 99)]
        print(np.nanmax(alive_values_flat), np.nanmax(dead_values_flat))
        # Plot
        plt.hist(alive_values_flat, bins=50, alpha=0.6, color='green', label='Healthy cells', density=True)
        plt.hist(dead_values_flat, bins=50, alpha=0.6, color='red', label='Dying cells', density=True)

        plt.xlabel(feature)
        plt.ylabel('Frequency Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_as)
    
    
def main():
    feature_names=[
        # 'Area',
        # 'Circularity',
        # 'Perimeter',
        # 'Displacement',
        # 'Mode 0',
        # 'Mode 1',
        # 'Mode 2',
        # 'Mode 3',
        # 'Mode 4',
        # 'Speed',
        # 'Phagocytes within 100 pixels',
        # 'Phagocytes within 250 pixels',
        # 'Phagocytes within 500 ',
        # 'X',
        # 'Y',
        'CellDeath',
        'Total Fluorescence', 
        'Fluorescence Distance Mean', 
        'Fluorescence Distance Variance'
    ]
    # plot_two_death_frame_hists(
    #     Path('temp') / 'death_frames_fine_tuned.txt',
    #     Path('temp') / 'cell_deaths_24_06.txt',
    #     label1='Fine Tuned',
    #     label2='Original',
    #     save_as=Path('temp') / 'death_frame_comparison.png'
    # )
    # plot_cell_features(1, 0, 50, Path('temp') / 'plot.png', feature_names=feature_names)
    # plot_average_cell_features(Path('temp') / 'plot.png', feature_names=feature_names)
    # plot_feature_correlations(Path('temp') / 'correlations_plot.png', feature_names=feature_names)
    # plot_num_dead_cells(Path('temp') / 'num_dead_cells_plot.png')
    # plot_feature_correlations(Path('temp') / 'plot.png', feature_names=[
    #     'Area',
    #     'Circularity',
    #     'Perimeter',
    #     'Displacement',
    #     'Mode 0',
    #     'Mode 1',
    #     'Mode 2',
    #     'Mode 3',
    #     'Speed'
    # ])
    # km_plot([Path('PhagoPred') / 'Datasets' / '16_09_1.h5',
    #         Path('PhagoPred') / 'Datasets' / '13_06_survival.h5',
    #         Path('PhagoPred') / 'Datasets' / '24_06_survival.h5'], 
    #         ['16_09', '13_06', '24_06'],
    #         Path('temp') / 'km_curve.png',
    #         (5, 1, 1))
    
    # compare_cell_features([Path('PhagoPred') / 'Datasets' / '13_06_survival.h5',
    #             Path('PhagoPred') / 'Datasets' / '24_06_survival.h5'], 
    #             ['13_06', '24_06'], 'Area',
    #             Path('temp') / 'features_plot.png')
    
    plot_alive_vs_dead_feature(Path('PhagoPred') / 'Datasets' / '24_06_survival.h5', 'Circularity',Path('temp') / 'features_plot.png')

if __name__ == '__main__':
    main()