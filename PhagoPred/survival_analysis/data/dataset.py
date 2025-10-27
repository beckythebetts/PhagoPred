from pathlib import Path
import threading

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy

from PhagoPred.feature_extraction.extract_features import CellType

class CellDataset(torch.utils.data.Dataset):
    def __init__(self,
                 hdf5_paths: list[Path],
                 features: list[str],
                #  pre_death_frames: int=0,
                 interpolate_nan: bool=False,
                 means: np.ndarray=None,
                 stds: np.ndarray=None,
                 specified_cell_idxs: list[int]=None,
                 event_time_bins = None,
                 num_bins: int = 16,
                 preload_data: bool = True,
                 uncensored_only: bool = False,
                 summary_stats: bool = False,
                 fixed_len: int = None,
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
            cell_idxs (list[int], optional): If specified, only these cell indices will be included.
            uncensored_only (bool): If true, only select cells for which time of death is known
        """
        self.hdf5_paths = hdf5_paths
        self.features = features
        self.cell_type = CellType('Phase')
        self.uncensored_only = uncensored_only
        self.summary_stats = summary_stats
        self.fixed_len = fixed_len
        # self.pre_death_frames = pre_death_frames
        self.interpolate_nan = interpolate_nan
        self.specified_cell_idxs = specified_cell_idxs if specified_cell_idxs and len(self.hdf5_paths) == 1 else None
        self.num_bins = num_bins

        self._lock = threading.Lock()
        self._files = [None] * len(hdf5_paths)  # To hold open file handles
        
        self._features_cache = [None] * len(hdf5_paths)  # Cache for features xr.Dataset

        self.event_time_bins = event_time_bins
        self.cell_metadata = None
        self._load_cell_metadata()

        # self.event_time_bins = self._compute_bins()

        self.means = means
        self.stds = stds
        
        if isinstance(self.means, torch.Tensor):
            self.means = self.means.cpu().numpy()
        if isinstance(self.stds, torch.Tensor):
            self.stds = self.stds.cpu().numpy()
        
        # self._preloaded_data = None
        if preload_data:
            self._preload_all_data()
        
        if summary_stats:
            self.summary_funcs = {
            'mean': lambda feats, diffs: np.nanmean(feats, axis=0),
            'std': lambda feats, diffs: np.nanstd(feats, axis=0),
            'skew': lambda feats, diffs: scipy.stats.skew(feats, axis=0, nan_policy='omit'),
            'total_ascent': lambda feats, diffs: np.nansum(diffs * (diffs > 0), axis=0),
            'total_descent': lambda feats, diffs: np.nansum(diffs * (diffs < 0), axis=0),
            'average_gradient': lambda feats, diffs: np.nansum(diffs, axis=0) / diffs.shape[0],
            'max': lambda feats, diffs: np.nanmax(feats, axis=0),
            'min': lambda feats, diffs: np.nanmin(feats, axis=0)
        }
            
    def _preload_all_data(self):
        """Load all cell features into RAM as numpy arrays."""
        for i, path in enumerate(self.hdf5_paths):
            print(f"Preloading data from file {i+1}/{len(self.hdf5_paths)}: {self.hdf5_paths[i]}")
            features_cache = self._get_features_np(i)
            self._features_cache[i] = features_cache
        if self.means is not None and self.stds is not None:
            self.normalise()
        print("Finished preloading all data into memory.")    
        
    def compute_normalisation_stats(self, sample_ratio=0.1):
        """Compute mean, std based on sampled data for large datasets."""
        sampled_data = []
        for path_idx, path in enumerate(self.hdf5_paths):
            f = self._get_file(path_idx)
            cell_features = self.cell_type.get_features_xr(f, features=self.features)
            cell_features = cell_features.to_dataarray(dim='Feature')
            sample_idxs = np.random.choice(cell_features.sizes['Cell Index'], size=int(cell_features.sizes['Cell Index'] * sample_ratio), replace=False)
            cell_features = cell_features.isel({'Cell Index': sample_idxs})
            cell_features = cell_features.transpose('Cell Index', 'Frame', 'Feature').to_numpy()  # shape: (num_cells, num_frames, num_features)
            cell_features = cell_features.reshape(-1, cell_features.shape[2]) #shape: (num_cells*num_frames, num_features)
            # print(cell_features.shape)
            sampled_data.append(cell_features)
        sampled_data = np.concatenate(sampled_data, axis=0)  # shape: (total num_cells*num_frames, num_features)
        # sampled_data = sampled_data.reshape(-1, sampled_data.shape[-1])  # shape: (total_num_cells * num_frames, num_features)
        means = np.nanmean(sampled_data, axis=0)
        stds = np.nanstd(sampled_data, axis=0)
        
        return means, stds
    
    def normalise(self):
        for file_idx, file_path in enumerate(self.hdf5_paths):
            self._features_cache[file_idx] = (self._features_cache[file_idx] - self.means[:,np.newaxis, np.newaxis]) / self.stds[:, np.newaxis, np.newaxis]
            
            
    def _compute_bins(self):
        obs_times = self.cell_metadata['End Frames'] - self.cell_metadata['Start Frames'] + 1
        # bins = np.quantile(obs_times, np.linspace(0, 1, self.num_bins + 1))
        bins = np.linspace(np.min(obs_times), np.max(obs_times), self.num_bins + 1)
        return bins
    
    def get_bins(self):
        if self.event_time_bins is None:
            self.event_time_bins = self._compute_bins()
            print("Computed event time bins.")
        return self.event_time_bins

    
    def get_normalization_stats(self):
        if self.means is None or self.stds is None:
            self.means, self.stds = self.compute_normalisation_stats()
            print("Computed normalisation statistics.")
            self.normalise()
        return self.means, self.stds
    
    def _get_file(self, idx):
        if self._files[idx] is None:
            with self._lock:
                if self._files[idx] is None:
                    self._files[idx] = h5py.File(self.hdf5_paths[idx], 'r')
                    
        return self._files[idx]
    
    # def _get_features_xr(self, file_idx):
    #     if self._features_cache[file_idx] is None:
    #         f = self._get_file(file_idx)
    #         self._features_cache[file_idx] = self.cell_type.get_features_xr(f, features=self.features)
    #     return self._features_cache[file_idx]
    
    def _get_features_np(self, file_idx: int, features: list = None) -> np.ndarray:
        if features is None:
            features = self.features
        feature_arrays = []
        for feature in features:
            f = self._get_file(file_idx)
            arr = f['Cells']['Phase'][feature][:]  # shape: (num_frame, num_cells)
            feature_arrays.append(arr)
        features_np = np.stack(feature_arrays, axis=0)
        return features_np
        
    def _load_cell_metadata(self):
        self.cell_metadata = {'File Idxs': [], 'Local Cell Idxs': [], 'Death Frames': [], 'Start Frames': [], 'End Frames': []}

        for path_idx, path in enumerate(self.hdf5_paths):
            total_cell_deaths = self._get_features_np(path_idx, features=['CellDeath'])[0][0] # shape = (num_cells)

            total_num_cells = len(total_cell_deaths)
            
            if self.uncensored_only:
                local_cell_idxs = np.nonzero(~np.isnan(total_cell_deaths))[0]
            else:
                local_cell_idxs = np.arange(total_num_cells)
                
            local_cell_idxs = local_cell_idxs.astype(int)

            cell_deaths = total_cell_deaths[local_cell_idxs]
            num_cells = len(cell_deaths)

            self.cell_metadata['File Idxs'] += [path_idx] * num_cells
            self.cell_metadata['Local Cell Idxs'] += list(local_cell_idxs)
            self.cell_metadata['Death Frames'] += list(cell_deaths)

            # Load only 'Area' to compute start/end frames
            # area_data = self.cell_type.get_features_xr(f, features=['Area'])['Area'].transpose('Cell Index', 'Frame').values[local_cell_idxs]  # shape: (num_cells, num_frames)
            area_data = self._get_features_np(path_idx, features=['Area'])
            area_data = area_data[0]
            area_data = area_data.T[local_cell_idxs]  # shape: (num_cells, num_frames)

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
        all_cell_features = self._features_cache[cell_metadata['File Idxs']] # shape: (feature, frame, cell)
        
        all_cell_features = all_cell_features.transpose(1, 2, 0) #shape: (frame, cell, feature)

        last_frame = cell_metadata['End Frames'] if np.isnan(cell_metadata['Death Frames']) else cell_metadata['Death Frames']
        event_indicator = 0 if np.isnan(cell_metadata['Death Frames']) else 1
        if self.fixed_len is None:
            landmark_frame = np.random.randint(cell_metadata['Start Frames'], last_frame)
            start_frame = cell_metadata['Start Frames']
        else:
            landmark_frame = np.random.randint(max(self.fixed_len, cell_metadata['Start Frame']))
            start_frame = last_frame - landmark_frame
        # start_frame = cell_metadata['Start Frames'] if self.fixed_len is None else last_frame - self.fixed_len

        cell_features = all_cell_features[start_frame:landmark_frame+1, cell_metadata['Local Cell Idxs']] # (num_frames, num_features)
        
        time_to_event = last_frame - landmark_frame
            
        time_to_event_bin = np.digitize(time_to_event, self.get_bins()) - 1  # Bin the observation time
        time_to_event_bin = np.clip(time_to_event_bin, 0, self.num_bins - 1)
        
        if self.summary_stats:
            cell_diffs = np.diff(cell_features, axis=0)
            results = []
            for summary_func in self.summary_funcs:
                results.append(summary_func(cell_features, cell_diffs))
            cell_features = np.stack(results, axis=0)

        return cell_features, time_to_event_bin, event_indicator, time_to_event

    def plot_event_vs_censoring_hist(self, title='Events vs Censored', save_path=None, bins=16, show=False):
        """
        Plot a histogram of observation times for event vs. censored samples.

        Args:
            save_path (str or Path, optional): Path to save the figure. If None, doesn't save.
            bins (int): Number of histogram bins.
            show (bool): Whether to display the plot.
        """
        observation_times = self.cell_metadata['End Frames'] - self.cell_metadata['Start Frames'] + 1
        death_frames = self.cell_metadata['Death Frames']
        
        is_event = ~np.isnan(death_frames)
        is_censored = np.isnan(death_frames)

        event_times = observation_times[is_event]
        censor_times = observation_times[is_censored]

        plt.figure(figsize=(10, 6))
        plt.hist(event_times, bins=bins, alpha=0.7, label=f'Deaths {len(event_times)}', color='tab:red')
        plt.hist(censor_times, bins=bins, alpha=0.7, label=f'Censored {len(censor_times)}', color='tab:blue')
        plt.xlabel('Observation Time (frames)')
        plt.ylabel('Number of Samples')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved event/censoring time histogram to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


# def summary_collate_fn(batch)

def collate_fn(batch, fixed_length=None, means=None, stds=None, device='cpu'):
    """
    Custom collate function to handle variable-length sequences and padding.
    """
    cell_features, time_to_event_bins, event_indicators, time_to_events = zip(*batch)
    # print(observation_times)

    time_to_event_bins = torch.tensor(time_to_event_bins, dtype=torch.long, device=device)
    event_indicators = torch.tensor(event_indicators, dtype=torch.float32, device=device)
    time_to_events = torch.tensor(time_to_events, dtype=torch.long, device=device)
    # print(observation_times)

    cell_features = [torch.tensor(features, dtype=torch.float32, device=device) for features in cell_features]  # List of [num_frames, num_features]
    # print(cell_features)

    if fixed_length is not None:
        cell_features = [f[:fixed_length] for f in cell_features]
        cell_features = [torch.cat([f, torch.zeros(fixed_length - len(f), f.shape[1])], dim=0) if len(f) < fixed_length else f for f in cell_features]
        lengths = torch.tensor([min(len(f), fixed_length) for f in cell_features], dtype=torch.long)
    else:
        lengths = torch.tensor([len(f) for f in cell_features], dtype=torch.long)
        cell_features = torch.nn.utils.rnn.pad_sequence(cell_features, batch_first=True, padding_value=0.0)  # [batch_size, max_seq_len, num_features]
        
    nan_mask = ~torch.isnan(cell_features)

    # Replace NaNs with 0
    cell_features = torch.where(nan_mask, cell_features, torch.tensor(0.0, device=device))

    # Normalize (broadcast means/stds to correct shape)
    # means and stds should be tensors of shape [num_features]
    # if means is not None and stds is not None:
    #     means = means.to(cell_features.device)
    #     stds = stds.to(cell_features.device)
    #     cell_features = (cell_features - means) / stds

    # Concatenate mask as additional features along last dimension
    cell_features = torch.cat([cell_features, nan_mask.float()], dim=-1)  # shape: [batch, frames, features*2]

    return cell_features, lengths, time_to_event_bins, event_indicators, time_to_events



    
