import torch
from pathlib import Path
import h5py
import numpy as np
import xarray as xr

from PhagoPred.feature_extraction.extract_features import CellType

class CellDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            hdf5_paths: list[Path], 
            fixed_length: int = None, 
            pre_death_frames: int = 0,
            include_alive: bool = True,
            num_alive_samples: int = 1,
            mode: str = 'train',
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
        self.hdf5_path = hdf5_paths
        self.fixed_length = fixed_length
        self.pre_death_frames = pre_death_frames
        self.include_alive = include_alive
        self.num_alive_samples = num_alive_samples
        self.mode = mode

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
        
        self.file_idxs, self.local_cell_idxs, self.death_frames = self._load_cells() 

        # self.file_cache = {}

    def _load_cells(self):
        """
        Load cells from the HDF5 files.

        Returns:
            list: List of cells loaded from the HDF5 file.
        """
        for path_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                cell_deaths = CellType('Phase').get_features_xr(f, features=['CellDeath'])['CellDeath'].sel(Frame=0).values
                cell_deaths = np.squeeze(cell_deaths)
                self.file_idxs = self.file_idxs + [path_idx] * len(cell_deaths)
                self.local_cell_idxs = self.local_cell_idxs + list(range(len(cell_deaths)))
                self.death_frames = self.death_frames + list(cell_deaths)

        self.file_idxs = np.array(self.file_idxs)
        self.local_cell_idxs = np.array(self.local_cell_idxs)
        self.death_frames = np.array(self.death_frames)

        return self.file_idxs, self.local_cell_idxs, self.death_frames
    
    def __len__(self):
        return len(self.file_idxs)
    
    def __getitems__(self, idxs):
        """
        Get items from the dataset based on the provided indices.
        Returns a list of tuples [(features, alive), ...] where features is a numpy array
        and alive is a boolean indicating if the cell is alive.
        """
        file_paths_idxs = self.file_idxs[idxs]
        local_cell_idxs = self.local_cell_idxs[idxs]
        death_frames = self.death_frames[idxs]

        cells = []

        for file_path_idx in np.unique(file_paths_idxs):
            file_cell_idxs = local_cell_idxs[file_paths_idxs == file_path_idx]
            file_death_frames = death_frames[file_paths_idxs == file_path_idx]
            with h5py.File(self.hdf5_paths[file_path_idx], 'r') as f:
                features_data = CellType('Phase').get_features_xr(f, features=self.features)
                for cell_idx, death_frame in zip(file_cell_idxs, file_death_frames):
                    cell_features = features_data.isel({'Cell Index': cell_idx})
                    cell_features = cell_features.to_array(dim='Feature')
                    if not np.isnan(death_frame):
                        end_frame = death_frame - self.pre_death_frames
                        if end_frame < 0:
                            continue
                    else:
                        if not self.include_alive:
                            continue
                        # last non-NaN frame is the end frame
                        end_frame = cell_features.sizes['Frame'] - 1 - (~np.isnan(cell_features)).any(dim='Feature')[::-1].argmax(dim='Frame')

                    if self.fixed_length is None:
                        # first non-NaN frame is the start frame
                        start_frame = (~np.isnan(cell_features)).any(dim='Feature').argmax(dim='Frame')
                    
                    else:
                        start_frame = end_frame - self.fixed_length + 1
                        if start_frame < 0:
                            continue
                    if start_frame > end_frame:
                        continue
                    features_data = features_data.isel({'Frame': slice(start_frame, end_frame+1), 'Cell Index': cell_idx})
                    cells.append((features_data.values, np.isnan(death_frame)))
    
        return cells

    def __getitem__(self, idx):
        """
        Get a single item from the dataset based on the provided index.
        Returns a tuple (features, alive) where features is a numpy array
        and alive is a boolean indicating if the cell is alive.
        """
        original_idx = idx
        while True:
            cell = self.__getitems__([idx])
            if cell:
                return cell[0]
            else:
                idx += 1
                if idx >= len(self):
                    idx = 0
                print(f"Warning: Empty sample at index {original_idx}, trying cell at index {idx}")
