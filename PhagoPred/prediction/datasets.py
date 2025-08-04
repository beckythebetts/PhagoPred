import warnings
from typing import Union

import torch
from pathlib import Path
import h5py
import numpy as np
import xarray as xr
import scipy
from tqdm import tqdm

from PhagoPred.feature_extraction.extract_features import CellType

class CellDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            hdf5_paths: list[Path], 
            fixed_length: int = None, 
            pre_death_frames: int = 0,
            include_alive: bool = True,
            num_alive_samples: int = 1,
            nan_threshold: float = 0.2,
            interpolate_nans: bool = True
    ):
        """
        Initialize the CellDataset.

        Args:
            hdf5_paths (Path): Path to the HDF5 file containing the dataset.
            fixed_length (int, optional): Fixed length of sequences. Defaults to None.
            pre_death_frames (int, optional): Number of frames before
                the death of the cell to include. Defaults to 0.
            include_alive (bool, optional): Whether to include alive cells in the dataset. Defaults to True.
            num_alive_samples (int, optional): Number of alive samples to include. Defaults to 1.
            mode (str, optional): Mode of the dataset ('train' or 'test'). Defaults to 'train'.
        """
        self.hdf5_paths = hdf5_paths
        self.fixed_length = fixed_length
        self.pre_death_frames = pre_death_frames
        self.include_alive = include_alive
        self.num_alive_samples = num_alive_samples
        self.nan_threshold = nan_threshold
        self.interpolate_nans = interpolate_nans

        self.features = [
            'Area',
            'Circularity',
            'Perimeter',
            'Displacement',
            'Mode 0',
            'Mode 1',
            'Mode 2',
            'Mode 3',
            'Mode 4',
            'Speed',
            'Phagocytes within 100 pixels',
            'Phagocytes within 250 pixels',
            'Phagocytes within 500 pixels',
            'Total Fluorescence', 
            'Fluorescence Distance Mean', 
            'Fluorescence Distance Variance',
            ]
        
        self.file_idxs, self.local_cell_idxs, self.death_frames, self.start_frames, self.end_frames = [], [], [], [], []
        self.file_idxs, self.local_cell_idxs, self.death_frames, self.start_frames, self.end_frames = self._load_cells() 


    def _load_cells(self):
        """
        Load cells from the HDF5 files and precompute start and end frames.

        Returns:
            tuple: file_idxs, local_cell_idxs, death_frames
        """
        self.start_frames = []
        self.end_frames = []

        for path_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                # Load death frames
                cell_deaths = CellType('Phase').get_features_xr(f, features=['CellDeath'])['CellDeath'].sel(Frame=0).values
                cell_deaths = np.squeeze(cell_deaths)
                num_cells = len(cell_deaths)

                self.file_idxs += [path_idx] * num_cells
                self.local_cell_idxs += list(range(num_cells))
                self.death_frames += list(cell_deaths)

                # Load only 'Area' to compute start/end frames
                area_data = CellType('Phase').get_features_xr(f, features=['Area'])['Area'].transpose('Cell Index', 'Frame').values  # shape: (num_cells, num_frames)

                # Use numpy vectorization for speed
                not_nan_mask = ~np.isnan(area_data)
                # First valid frame (start frame)
                start = not_nan_mask.argmax(axis=1)
                # Last valid frame (end frame)
                reversed_mask = not_nan_mask[:, ::-1]
                last = area_data.shape[1] - 1 - reversed_mask.argmax(axis=1)

                self.start_frames += list(start)
                self.end_frames += list(last)

        self.file_idxs = np.array(self.file_idxs)
        self.local_cell_idxs = np.array(self.local_cell_idxs)
        self.death_frames = np.array(self.death_frames)
        self.start_frames = np.array(self.start_frames)
        self.end_frames = np.array(self.end_frames)

        # Compute valid indices using your existing method
        valid_idxs = self._compute_valid_idxs()

        # Filter all arrays to keep only valid indices
        self.file_idxs = self.file_idxs[valid_idxs]
        self.local_cell_idxs = self.local_cell_idxs[valid_idxs]
        self.death_frames = self.death_frames[valid_idxs]
        self.start_frames = self.start_frames[valid_idxs]
        self.end_frames = self.end_frames[valid_idxs]

        return self.file_idxs, self.local_cell_idxs, self.death_frames, self.start_frames, self.end_frames
    
    def _compute_valid_idxs(self):
        # store interpolate_nans value
        store_interpolate_nan = self.interpolate_nans

        # set interpolate nans to false, don't need to be calucalted now
        self.interpolate_nans = False

        valid_idxs = []
        batch_size = 1000
        print("\n=== Checking for valid samples ===\n")
        for start in range(0, len(self), batch_size):
            batch_indices = np.arange(start, min(start + batch_size, len(self)))
            batch_samples = CellDataset.__getitems__(self, batch_indices)
            for idx, sample in zip(batch_indices, batch_samples):
                # if sample != (None, None):
                if sample[0] is not None and sample[1] is not None:
                    valid_idxs.append(idx)


        # set interpolate_nans back to stored value
        self.interpolate_nans = store_interpolate_nan
        return valid_idxs

    def __len__(self):
        return len(self.file_idxs)
    
    def __getitems__(self, idxs: list[int]):
        """
        Get items from the dataset based on the provided indices.
        Returns a list of tuples [(features, alive), ...] where features is a numpy array
        and alive is a boolean indicating if the cell is alive.
        """
        file_paths_idxs = self.file_idxs[idxs]
        local_cell_idxs = self.local_cell_idxs[idxs]
        death_frames = self.death_frames[idxs]
        start_frames = self.start_frames[idxs]
        end_frames = self.end_frames[idxs]

        cells = []

        for file_path_idx in np.unique(file_paths_idxs):
            file_cell_idxs = local_cell_idxs[file_paths_idxs == file_path_idx]
            file_death_frames = death_frames[file_paths_idxs == file_path_idx]
            file_start_frames = start_frames[file_paths_idxs == file_path_idx]
            file_end_frames = end_frames[file_paths_idxs == file_path_idx]


            with h5py.File(self.hdf5_paths[file_path_idx], 'r') as f:
                features_data = CellType('Phase').get_features_xr(f, features=self.features)

                for cell_idx, death_frame, start_frame, end_frame in tqdm(zip(
                    file_cell_idxs, 
                    file_death_frames,
                    file_start_frames, 
                    file_end_frames,
                    ), desc=f"Getting cells from file {self.hdf5_paths[file_path_idx].name}", total=len(file_cell_idxs)):

                    cell_features = features_data.isel({'Cell Index': cell_idx})
                    cell_features = cell_features.to_dataarray(dim='Feature').transpose()

                    if not np.isnan(death_frame):
                        end_frame = death_frame - self.pre_death_frames
                        if end_frame < 0:
                            cells.append((None, None))
                            continue
                    else:
                        if not self.include_alive:
                            cells.append((None, None))
                            continue

                    if self.fixed_length is not None:
                        start_frame = int(end_frame) - int(self.fixed_length) + 1
                        if start_frame < 0:
                            cells.append((None, None))
                            continue
                    if start_frame > end_frame:
                        cells.append((None, None))
                        continue
                    cell_features_data = cell_features.isel({'Frame': slice(int(start_frame), int(end_frame+1))})

                    features_array = cell_features_data.values  # (Features, Frames)

                    # Compute proportion of NaNs
                    nan_ratio = np.isnan(features_array).sum() / features_array.size
                    if nan_ratio > self.nan_threshold:
                        cells.append((None, None))
                        continue  # Skip this cell

                    if self.interpolate_nans:
                        features_array = self.apply_interpolate_nans(features_array)

                    cells.append((features_array, np.isnan(death_frame)))    
        return cells

    def __getitem__(self, idx: Union[int, list[int]]):
        """
        Get a single item from the dataset based on the provided index.
        Returns a tuple (features, alive) where features is a numpy array
        and alive is a boolean indicating if the cell is alive.
        """
        if isinstance(idx, (list, np.ndarray)):
            return self.__getitems__(idx)
        else:
            cell = self.__getitems__([idx])
            return cell[0]
        
    def apply_interpolate_nans(self, data):
        # data shape: (frames, features)
        for feature_idx in range(data.shape[1]):
            feature_data = data[:, feature_idx]
            nans = np.isnan(feature_data)
            if np.any(nans):
                not_nans = ~nans
                x = np.arange(len(feature_data))
                feature_data[nans] = np.interp(x[nans], x[not_nans], feature_data[not_nans])
                data[:, feature_idx] = feature_data
        return data


class SummaryStatsCellDataset(CellDataset):
    """
    Dataset that computes summary statistics for each cell.
    """
    def __init__(self, *args, **kwargs):
        
        self.stat_functions = {
            'mean': lambda feats, diffs: np.nanmean(feats, axis=0),
            'std': lambda feats, diffs: np.nanstd(feats, axis=0),
            'skew': lambda feats, diffs: scipy.stats.skew(feats, axis=0, nan_policy='omit'),
            'total_ascent': lambda feats, diffs: np.nansum(diffs * (diffs > 0), axis=0),
            'total_descent': lambda feats, diffs: np.nansum(diffs * (diffs < 0), axis=0),
            'average_gradient': lambda feats, diffs: np.nansum(diffs, axis=0) / diffs.shape[0],
            'max': lambda feats, diffs: np.nanmax(feats, axis=0),
            'min': lambda feats, diffs: np.nanmin(feats, axis=0)
        }

        

        super().__init__(*args, **kwargs)

        self.feature_names = [f"{feature}_{stat_name}" for stat_name in self.stat_functions.keys() for feature in self.features]

        self.interpolate_nans = False
    
    def get_summary_stats(self, features):
        """
        Compute summary statistics for the given features.

        Args:
            features (numpy.ndarray): Features of the cell.

        Returns:
            numpy.ndarray: Summary statistics.
        """
        stats = []
        diffs = np.diff(features, n=1, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Ignore warnings about NaN in mean/std/skew calculations
            # This is because we may have NaN values in the features
            # and we want to compute stats ignoring those NaNs.
            for func in self.stat_functions.values():
                stat = func(features, diffs)
                if isinstance(stat, np.ndarray):
                    stats.append(stat)
                else:
                    stats.append(np.array([stat]))
        return np.concatenate(stats, axis=0)
    
    def __getitems__(self, idxs):
        """
        Get items from the dataset based on the provided indices.
        Returns a list of tuples [(summary_stats, alive), ...] where summary_stats is a numpy array
        and alive is a boolean indicating if the cell is alive.
        """
        cells = super().__getitems__(idxs)
        summary_stats = []
        for features, alive in cells:
            stats = self.get_summary_stats(features)
            summary_stats.append((stats.flatten(), alive))
        return summary_stats