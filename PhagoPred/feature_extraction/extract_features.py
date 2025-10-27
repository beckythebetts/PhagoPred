import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import pandas as pd
import xarray as xr
import dask.array as da
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction import features
from PhagoPred.utils import tools


class CellType:
    FRAME_DIM = 0
    CELL_DIM = 1

    DIMS = ["Frame", "Cell Index"]

    def __init__(self, name: str) -> None:

        self.name = name

        self.features_group = f'Cells/{name}'
        self.images = f'Images/{name}'
        self.masks = f'Segmentations/{name}'

        self.primary_features = []
        self.derived_features = []
        self.primary_derived_features = []
        
        self.feature_names = []

        self.primary_feature_names = []
        self.derived_feature_names = []
        self.primary_derived_feature_names = []

        self.primary_feature_datasets = []
        self.derived_feature_datasets = []
        self.primary_derived_feature_datasets = []

        self.num_cells = None


    def set_start_end_death_frames(self):
        """Add atributes  to self.features_group consisting of lists of frames of first appearnance, last appearance and death for each cell.
        For no cell death observecm dell death will be -1."""

    def add_feature(self, feature):
        if feature.primary_feature:
            if feature.derived_feature:
                self.primary_derived_features.append(feature)
            else:   
                self.primary_features.append(feature)
        else:
            self.derived_features.append(feature)

    def get_feature_names(self):

        for feature in self.primary_features:
            for name in feature.get_names():
                self.primary_feature_names.append(name)

        for feature in self.derived_features:
            for name in feature.get_names():
                self.derived_feature_names.append(name)

        for feature in self.primary_derived_features:
            for name in feature.get_names():
                self.primary_derived_feature_names.append(name)

        self.feature_names = self.primary_feature_names + self.derived_feature_names + self.primary_derived_feature_names
        return self.feature_names
    

    def get_masks(self, h5py_file: h5py.File, first_frame: int, last_frame: int = None) -> np.array:
        """
        Gets masks between the first and last frame, assuming h5py file is open and readable.
        masks dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1

        masks = h5py_file[self.masks][first_frame:last_frame]
        if len(masks) == 1:
            return masks[0]
        else:
            return masks
    

    def get_images(self, h5py_file: h5py.File, first_frame: int, last_frame: int = None) -> np.array:
        """
        Gets images between the first and last frame, assuming h5py file is open and readable.
        dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1

        images = h5py_file[self.images][first_frame:last_frame]

        if len(images) == 1:
            return images[0]
        
        else:
            return images

    def set_num_cells(self, num_cells):
        self.num_cells = num_cells


    def set_up_features_group(self, h5py_file: h5py.File):
        """
        Create hdf5 datset for each feature
        """
        num_frames = h5py_file[self.images].shape[self.FRAME_DIM]

        num_cells = 1

        if 'X' in h5py_file[self.features_group].keys():
            num_cells = h5py_file[self.features_group]['X'].shape[1]
        for feature_name in self.primary_feature_names:
            dataset = h5py_file.require_dataset(f'{self.features_group}/{feature_name}', 
                                                shape=(num_frames, num_cells), 
                                                maxshape=(num_frames, None), 
                                                dtype=np.float32,
                                                fillvalue=np.nan, 
                                                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.primary_feature_datasets.append(dataset)

        for feature_name in self.derived_feature_names:
            dataset = h5py_file.require_dataset(f'{self.features_group}/{feature_name}', 
                                                shape=(num_frames, num_cells), 
                                                maxshape=(num_frames, None), 
                                                dtype=np.float32,
                                                fillvalue=np.nan,
                                                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.derived_feature_datasets.append(dataset)

        for feature_name in self.primary_derived_feature_names:
            dataset = h5py_file.require_dataset(f'{self.features_group}/{feature_name}', 
                                                shape=(num_frames, num_cells), 
                                                maxshape=(num_frames, None), 
                                                dtype=np.float32,
                                                fillvalue=np.nan,
                                                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.primary_derived_feature_datasets.append(dataset)


    def get_features_xr(self, h5py_file: h5py.File, features: list = None) -> xr.Dataset:
        """
        h5py file is open file, returns x_array (chunked like h5py). If features not specified, reads all features iwth both cell index and time dimensions.
        """
        data_dict = {}
        for feature_name, feature_data in h5py_file[self.features_group].items():
            if features:
                if feature_name not in features:
                    continue
            else:
                if feature_data.shape[0] == 1:
                    continue
            # read as dask array, preserving chunks
            data = da.from_array(feature_data, chunks=feature_data.chunks)
            data_dict[feature_name] = (self.DIMS, data)

        ds = xr.Dataset(data_dict)

        # ds = ds.assign_coords(Frame=np.arange(ds.sizes['Frame']))
        
        return ds

    
class FeaturesExtraction:
 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self, 
            h5py_file=SETTINGS.DATASET, 
            frame_batchsize=10,
            cell_batch_size=100
            ) -> None:
        
        self.num_frames = None
        
        self.h5py_file = h5py_file
        self.frame_batchsize = frame_batchsize
        self.cell_batch_size = cell_batch_size

        self.cell_types = [CellType('Phase')]
        self.cell_type_names = [cell_type.name for cell_type in self.cell_types]

        self.set_up()


    def get_cell_type(self, cell_type_name: str):
        return self.cell_types[self.cell_type_names.index(cell_type_name)]
    

    def add_feature(self, feature, cell_type: str) -> None:
        """
        Add feature to specified cell type
        """
        self.cell_types[self.cell_type_names.index(cell_type)].add_feature(feature)


    def set_up(self):
        """
        Prepare hdf5 file
        """
        with h5py.File(self.h5py_file, 'r+') as f:
            # self.num_frames = SETTINGS.NUM_FRAMES
            self.num_frames = f['Images']['Phase'].shape[0]
            for cell_type in self.cell_types:
                cell_type.get_feature_names()
                cell_type.set_up_features_group(f)
                

    def extract_features(self) -> None:
        with h5py.File(self.h5py_file, 'r+') as f:
            for cell_type in self.cell_types:
                self.extract_primary_features(f, cell_type)
                self.extract_primary_derived_features(f, cell_type)
                self.extract_derived_features(f, cell_type)
            

    def extract_primary_features(self, f: h5py.File, cell_type: CellType) -> None:
        if len(cell_type.primary_features) > 0:
            print(f'\n=== Calculating Primary Features ({cell_type.name}) ===\n')
            # phase_xr = self.cell_types[self.cell_type_names.index('Phase')].get_features_xr(f)
            # epi_xr = None
            # if 'Epi' in self.cell_type_names:   
            #     epi_xr = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_xr(f)

            for frame_idx in tqdm(range(self.num_frames)):

                # sys.stdout.write(f'\r=== Calculating Primary Features ({cell_type.name}) ===')
                # sys.stdout.flush()

                mask = cell_type.get_masks(f, frame_idx)
                image = cell_type.get_images(f, frame_idx)

                num_cells = np.max(mask) + 1

                for first_cell in range(0, num_cells, self.cell_batch_size):

                    last_cell = min(first_cell + self.cell_batch_size, num_cells)

                    cell_idxs = torch.arange(first_cell, last_cell).to(self.DEVICE)

                    expanded_mask = torch.tensor(mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                    for feature in cell_type.primary_features:
                        result = feature.compute(mask=expanded_mask, image=image)
                        # print(result.shape)
                        if result.ndim == 1:
                            result = result[:, np.newaxis]
                        for i, feature_name in enumerate(feature.get_names()):

                            #resize dataset if too many cells
                            if num_cells>f[cell_type.features_group][feature_name].shape[cell_type.CELL_DIM]:
                                f[cell_type.features_group][feature_name].resize(num_cells, cell_type.CELL_DIM)

                            f[cell_type.features_group][feature_name][frame_idx, first_cell:last_cell] = result[:, i]
        
                    torch.cuda.empty_cache()
    
    def extract_primary_derived_features(self, f: h5py.File, cell_type: CellType) -> None:
        """
        Extracts primary and derived features for a given cell type.
        """
        # self.extract_primary_features(f, cell_type)
        
        if len(cell_type.primary_derived_features) > 0:
            print(f'\n=== Calculating Primary/Derived Features ({cell_type.name}) ===\n')
            phase_features_xr = self.cell_types[self.cell_type_names.index('Phase')].get_features_xr(f)
            epi_features_xr = None

            if 'Epi' in self.cell_type_names:
                epi_features_xr = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_xr(f)
            
            for frame_idx in tqdm(range(self.num_frames)):
                mask = cell_type.get_masks(f, frame_idx)
                image = cell_type.get_images(f, frame_idx)
                epi_image = CellType('Epi').get_images(f, frame_idx)

                frame_phase_xr = phase_features_xr.isel(Frame=frame_idx)
                frame_epi_xr = None
                
                num_cells = np.max(mask) + 1

                for first_cell in range(0, num_cells, self.cell_batch_size):

                    last_cell = min(first_cell + self.cell_batch_size, num_cells)

                    cell_idxs = np.arange(first_cell, last_cell)
                    phase_xr = frame_phase_xr.isel({'Cell Index': cell_idxs})

                    cell_idxs = torch.from_numpy(cell_idxs).to(self.DEVICE)

                    expanded_mask = torch.tensor(mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                    for feature in cell_type.primary_derived_features:
                        result = feature.compute(mask=expanded_mask, image=image, epi_image=epi_image, phase_xr=phase_xr, epi_xr=None)
                        if result.ndim == 1:
                            result = result[:, np.newaxis]
                        for i, feature_name in enumerate(feature.get_names()):

                            #resize dataset if too many cells
                            if num_cells>f[cell_type.features_group][feature_name].shape[cell_type.CELL_DIM]:
                                f[cell_type.features_group][feature_name].resize(num_cells, cell_type.CELL_DIM)

                            f[cell_type.features_group][feature_name][frame_idx, first_cell:last_cell] = result[:, i]
        
                    torch.cuda.empty_cache()
                


    def extract_derived_features(self, f: h5py.File, cell_type: CellType) -> None:
        if len(cell_type.derived_features) > 0:
            print(f'\n=== Calculating derived features ({cell_type.name}) ===\n')

            phase_features_xr = self.cell_types[self.cell_type_names.index('Phase')].get_features_xr(f)

            epi_features_xr = None
            if 'Epi' in self.cell_type_names:
                epi_features_xr = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_xr(f)

            for feature in cell_type.derived_features:
                print(f'Calculating {", ".join(feature.get_names())}...')
                result = feature.compute(phase_xr=phase_features_xr, epi_xr=epi_features_xr)

                if result.ndim == 2:
                    result = result[:, :, np.newaxis]
                
                for i, feature_name in enumerate(feature.get_names()):
                    if result.shape[cell_type.FRAME_DIM] == 1:
                        f[cell_type.features_group][feature_name].resize(1, cell_type.FRAME_DIM)
                    f[cell_type.features_group][feature_name][:] = result[:, :, i]

# def cell_death_hyperparamter_search(dataset: Path = SETTINGS.DATASET, true_deaths_txt: Path = None) -> None:
#     # with open(true_deaths_txt, 'r') as f:
#     true_deaths = np.genfromtxt(true_deaths_txt, delimiter=', ', usecols=1)
    
#     thresholds = np.arange(0.2, 0.7, 0.1)
#     min_frames_deads = np.arange(5, 50, 5)
#     smoothed_frames = np.arange(3, 20, 2)
#     for threshold in thresholds:
#         for min_frames_dead in min_frames_deads:
#             for smoothed_frame in smoothed_frames:
#                 extract_features(dataset, phase_features=features.CellDeath(threshold, min_frames_dead, smoothed_frame))
#                 with h5py.File(dataset, 'r') as f:
#                     estimated_deaths = f['Cells']['Phase']['CellDeath'][0, :len(true_deaths)]
                
#     accuracy = 
#     print(true_deaths, estimated_deaths)
def cell_death_hyperparameter_search(dataset: Path, true_deaths_txt: Path, save_csv: bool = False, csv_path: Path = None) -> pd.DataFrame:
    # Load true deaths (NaNs preserved)
    true_deaths = np.genfromtxt(true_deaths_txt, delimiter=',', usecols=1)

    thresholds = np.arange(0.01, 0.3, 0.05)
    smoothed_frames = np.array([1, 2, 3])
    min_frames_deads = np.arange(5, 30, 5)

    results = []

    # Optional: show progress bar
    total_iters = len(thresholds) * len(min_frames_deads) * len(smoothed_frames)
    pbar = tqdm(total=total_iters, desc="Hyperparam search")

    for threshold in thresholds:
        for min_frames_dead in min_frames_deads:
            for smoothed_frame in smoothed_frames:
                # Call your extraction function that updates the dataset with the predicted deaths
                extract_features(dataset, phase_features=[features.CellDeath(threshold, min_frames_dead, smoothed_frame)])

                with h5py.File(dataset, 'r') as f:
                    estimated_deaths = f['Cells']['Phase']['CellDeath'][0, :len(true_deaths)]

                has_death_mask = ~np.isnan(true_deaths)
                no_death_mask = np.isnan(true_deaths)

                # Compute MAE on cells with true death frames and predicted death frames
                valid_mask = has_death_mask & ~np.isnan(estimated_deaths)
                mae = np.nan
                if np.any(valid_mask):
                    mae = np.mean(np.abs(estimated_deaths[valid_mask] - true_deaths[valid_mask]))

                # Compute accuracy for no-death cells (fraction of predicted NaNs)
                no_death_accuracy = np.nan
            

                # Create binary labels
                true_labels = ~np.isnan(true_deaths)           # True = death, False = no-death
                pred_labels = ~np.isnan(estimated_deaths)      # True = predicted death

                # Check if there are any labels at all to evaluate
                if np.any(true_labels) or np.any(~true_labels):
                    precision = precision_score(true_labels, pred_labels, zero_division=0)
                    recall = recall_score(true_labels, pred_labels, zero_division=0)
                    f1 = f1_score(true_labels, pred_labels, zero_division=0)
                else:
                    precision = recall = f1 = np.nan  # No data to compute on

                results.append({
                    'threshold': threshold,
                    'min_frames_dead': min_frames_dead,
                    'smoothed_frame': smoothed_frame,
                    'mae_death_cells': mae,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                print(mae, f1)

                pbar.update(1)

    pbar.close()

    results_df = pd.DataFrame(results)

    if save_csv:
        if csv_path is None:
            csv_path = Path("cell_death_hyperparam_search_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return results_df

import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path

def plot_hyperparam_heatmaps(results_path_or_df, save_dir='hyperparam_heatmaps'):
    # Load results
    if isinstance(results_path_or_df, pd.DataFrame):
        df = results_path_or_df
    else:
        df = pd.read_csv(results_path_or_df)

    print(df.dtypes)
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print best rows for each metric
    print("Best row by MAE (lowest):")
    print(df.loc[df['mae_death_cells'].idxmin()])
    print("\nBest row by No-death F1-Score (highest):")
    print(df.loc[df['f1_score'].idxmax()])

    # List of hyperparameters (exclude performance columns)
    # param_cols = [c for c in df.columns if c not in ['mae_death_cells', 'accuracy_no_death_cells']]
    param_cols = [c for c in df.columns if c not in ['mae_death_cells', 'f1_score', 'precision', 'recall']]
    # For every pair of hyperparameters, plot heatmaps for MAE and Accuracy
    for x_param, y_param in combinations(param_cols, 2):
        # Pivot tables for heatmaps (mean if multiple entries)
        mae_pivot = df.pivot_table(index=y_param, columns=x_param, values='mae_death_cells', aggfunc='mean')
        acc_pivot = df.pivot_table(index=y_param, columns=x_param, values='f1_score', aggfunc='mean')

        # Plot MAE heatmap (lower is better)
        plt.figure(figsize=(8, 6))
        # sns.heatmap(mae_pivot, annot=True, fmt=".3f", cmap='viridis_r', cbar_kws={'label': 'MAE Death Cells'})
        plt.title(f'MAE Death Cells: {y_param} vs {x_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.tight_layout()
        plt.savefig(save_dir / f'mae_{y_param}_vs_{x_param}.png')
        plt.close()

        # Plot No-Death Accuracy heatmap (higher is better)
        plt.figure(figsize=(8, 6))
        # sns.heatmap(acc_pivot, annot=True, fmt=".3f", cmap='viridis', cbar_kws={'label': 'Accuracy No-Death Cells'})
        plt.title(f'Accuracy No-Death Cells: {y_param} vs {x_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.tight_layout()
        plt.savefig(save_dir / f'accuracy_{y_param}_vs_{x_param}.png')
        plt.close()

    print(f"Heatmaps saved to {save_dir.resolve()}")
    
    
def extract_features(dataset=SETTINGS.DATASET, 
                     phase_features = [
                        features.Fluorescence(), 
                        # features.MorphologyModes(), 
                        features.Speed(),
                        features.DensityPhase(),
                        features.Displacement(),
                        features.Perimeter(),
                        features.Circularity(),
                        # features.Skeleton(),
                        # features.UmapEmbedding(),
                        # features.GaborScale(),
                        features.CellDeath(),
                        features.FirstLastFrame(),
                        ]):
    
    feature_extractor = FeaturesExtraction(h5py_file=dataset)
    for feature in phase_features:
        feature_extractor.add_feature(feature, 'Phase')

    feature_extractor.set_up()
    feature_extractor.extract_features()

# def extract_and_clean(dataset=SETTINGS.DATASET):
#     extract_features(dataset, features.Perimeter())
#     clean_features.remove_bad_frames(dataset, use_features=['Area', 'Perimeter'])
#     extract_features(dataset)
    
def main():
    # cell_death_hyperparamter_search(true_deaths_txt='/home/ubuntu/PhagoPred/temp/03_10_deaths.txt')
    cell_death_hyperparameter_search(dataset=SETTINGS.DATASET, true_deaths_txt='/home/ubuntu/PhagoPred/temp/03_10_deaths.txt', save_csv=True, csv_path = '/home/ubuntu/PhagoPred/temp/death_hyperparamters.txt')
    plot_hyperparam_heatmaps('/home/ubuntu/PhagoPred/temp/death_hyperparamters.txt', '/home/ubuntu/PhagoPred/temp/death_hyperparamters')
    # extract_features(phase_features=[features.CellDeath()])
    # with h5py.File('PhagoPred/Datasets/16_09_1.h5', 'r') as f:
    #     print(np.nanmax(f['Cells']['Phase']['Circularity'][:]))
    #     smallest = np.nanargmin(f['Cells']['Phase']['Perimeter'][:])
    #     frame, cell = np.unravel_index(smallest, f['Cells']['Phase']['Area'].shape)
    #     print(frame, cell)

if __name__ == '__main__':
    main()
        