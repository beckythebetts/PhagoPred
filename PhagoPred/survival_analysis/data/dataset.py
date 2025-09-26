from pathlib import Path

import torch
import h5py
import numpy as np

from PhagoPred.feature_extraction.extract_features import CellType

class CellDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_paths: list[Path],
                 features: list[str],
                 pre_death_frames: int=0,
                 interpolate_nan: bool=False,
                 ):
        """
        A PyTorch Dataset for loading cell data from multiple HDF5 files.
        To be used for survival analysis models with time varying covariates, e.g Dynamic DeepHit.
        Args:
            hdf5_paths (list[Path]): List of paths to HDF5 files containing cell data.
            features (list[str]): List of feature names to be used.
            fixed_length (int, optional): If specified, all sequences will be padded or truncated to this length.
            pre_death_frames (int, optional): Number of frames before cell death
            nan_threshold (float, optional): Maximum allowed fraction of NaN values in a sequence.
        """
        self.hdf5_paths = hdf5_paths
        self.features = features
        self.pre_death_frames = pre_death_frames
        self.interpolate_nan = interpolate_nan

        self.cell_metadata = None
        self._load_cell_metadata()

        self.means, self.stds = self.compute_normalization_stats()

        print(f"Loaded dataset from {', '.join(map(str, hdf5_paths))}")

    def compute_normalization_stats(self):
        all_data = []
        for path in self.hdf5_paths:
            with h5py.File(path, 'r') as f:
                cell_features = CellType('Phase').get_features_xr(f, features=self.features)
                cell_features = cell_features.to_dataarray(dim='Feature')
                cell_features = cell_features.transpose('Cell Index', 'Frame', 'Feature').to_numpy()  # shape: (num_cells, num_frames, num_features)
                all_data.append(cell_features)
        all_data = np.concatenate(all_data, axis=0)  # shape: (total_num_cells, num_frames, num_features)
        all_data = all_data.reshape(-1, all_data.shape[-1])  # shape: (total_num_cells * num_frames, num_features)
        means = np.nanmean(all_data, axis=0)
        stds = np.nanstd(all_data, axis=0)
        return means, stds
    
    def _load_cell_metadata(self):
        self.cell_metadata = {'File Idxs': [], 'Local Cell Idxs': [], 'Death Frames': [], 'Start Frames': [], 'End Frames': []}

        for path_idx, path in enumerate(self.hdf5_paths):
            with h5py.File(path, 'r') as f:
                # Load death frames
                cell_deaths = CellType('Phase').get_features_xr(f, features=['CellDeath'])['CellDeath'].sel(Frame=0).values
                cell_deaths = np.squeeze(cell_deaths)
                num_cells = len(cell_deaths)

                self.cell_metadata['File Idxs'] += [path_idx] * num_cells
                self.cell_metadata['Local Cell Idxs'] += list(range(num_cells))
                self.cell_metadata['Death Frames'] += list(cell_deaths)

                # Load only 'Area' to compute start/end frames
                area_data = CellType('Phase').get_features_xr(f, features=['Area'])['Area'].transpose('Cell Index', 'Frame').values  # shape: (num_cells, num_frames)

                # Use numpy vectorization for speed
                not_nan_mask = ~np.isnan(area_data)
                # First valid frame (start frame)
                start = not_nan_mask.argmax(axis=1)
                # Last valid frame (end frame)
                reversed_mask = not_nan_mask[:, ::-1]
                last = area_data.shape[1] - 1 - reversed_mask.argmax(axis=1)

                self.cell_metadata['Start Frames'] += list(start)
                self.cell_metadata['End Frames'] += list(last)
     
            for key, value in self.cell_metadata.items():
                self.cell_metadata[key] = np.array(value)

    def __len__(self):
        return len(self.cell_metadata['File Idxs'])

    def __getitem__(self, idx):
        """
        Returns (features, observation_time, event_indicator)
        """
        cell_metadata = {key: self.cell_metadata[key][idx] for key in self.cell_metadata}
        with h5py.File(self.hdf5_paths[cell_metadata['File Idxs']], 'r') as f:
            all_cell_features = CellType('Phase').get_features_xr(f, features=self.features)
            
            observation_time = cell_metadata['End Frames'] - cell_metadata['Start Frames'] + 1
            event_indicator = 0 if np.isnan(cell_metadata['Death Frames']) else 1

            cell_features = all_cell_features.isel({'Cell Index': cell_metadata['Local Cell Idxs']})
            cell_features = cell_features.sel(Frame=slice(cell_metadata['Start Frames'], cell_metadata['End Frames'] - self.pre_death_frames if event_indicator == 1 else cell_metadata['End Frames']))
            cell_features = cell_features.to_dataarray(dim='Feature')
            cell_features = cell_features.transpose('Frame', 'Feature').to_numpy()  # [num_frames, num_features]
            
            nan_mask = ~np.isnan(cell_features)
            cell_features = np.nan_to_num(cell_features, nan=0.0)
            cell_features = (cell_features - self.means) / self.stds

            cell_features = np.concatenate([cell_features, nan_mask.astype(np.float32)], axis=-1)  # Add mask as additional features
        
        return cell_features, observation_time, event_indicator
    

def collate_fn(batch, fixed_length=None):
    """
    Custom collate function to handle variable-length sequences and padding.
    """
    cell_features, observation_times, event_indicators = zip(*batch)

    observation_times = torch.tensor(observation_times, dtype=torch.float32)
    event_indicators = torch.tensor(event_indicators, dtype=torch.float32)

    cell_features = [torch.tensor(features, dtype=torch.float32) for features in cell_features]  # List of [num_frames, num_features]

    if fixed_length is not None:
        cell_features = [f[:fixed_length] for f in cell_features]
        cell_features = [torch.cat([f, torch.zeros(fixed_length - len(f), f.shape[1])], dim=0) if len(f) < fixed_length else f for f in cell_features]
        lengths = torch.tensor([min(len(f), fixed_length) for f in cell_features], dtype=torch.long)
    else:
        lengths = torch.tensor([len(f) for f in cell_features], dtype=torch.long)
        cell_features = torch.nn.utils.rnn.pad_sequence(cell_features, batch_first=True, padding_value=0.0)  # [batch_size, max_seq_len, num_features]

    return cell_features, lengths, observation_times, event_indicators
