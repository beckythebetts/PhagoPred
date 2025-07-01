import sys
import gc

from typing import Optional, Tuple
import torch
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
import xarray as xr

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features
from PhagoPred.utils import tools

class Tracker:

    def __init__(self, file: Path = SETTINGS.DATASET, channel: str = 'Phase', frame_batch_size: int = 20) -> None:
        if file is not None:
            self.file = file
            self.channel = channel
            self.frame_batch_size = frame_batch_size

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = extract_features.FeaturesExtraction(self.file)
            self.cell_type = self.feature_extractor.get_cell_type(self.channel)
            self.masks_ds = f'Segmentations/{self.channel}'
            self.cells_group = f'Cells/{self.channel}'

            self.current_frames = slice(None)

            self.old_cells = None

    def track(self):
        self.get_cell_info()
        for first_frame in range(0, SETTINGS.NUM_FRAMES, self.frame_batch_size):
            last_frame = min(first_frame + self.frame_batch_size, SETTINGS.NUM_FRAMES)
            self.current_frames = slice(first_frame, last_frame)
            self.get_tracklets()
            self.join_tracklets()
            self.delete_nan_cells()
        
        self.current_frames = slice(None)
        self.join_tracklets()
        self.remove_short_tracks()
        self.delete_nan_cells()


    def get_cell_info(self):
        """
        Add each cells centroid cooridnates and areas to cells dataset
        """
        self.feature_extractor.add_feature(features.Coords(), cell_type=self.channel)
        self.feature_extractor.set_up()
        self.feature_extractor.extract_features()

    

    def get_tracklets(self, max_dist: int = SETTINGS.MAXIMUM_DISTANCE_THRESHOLD) -> None:
        """Between each pair of consecutive frames, reassign the cell indexes of the second frame to minimise the total distance gap between all cells centroids.
        To join two cells, their distance gap must be less than max_dist
        """
        with h5py.File(self.file, 'r+') as f:
            f[self.cells_group].attrs['maximum cell distance moved between frames'] = max_dist
            
            coords_list = ['X', 'Y']
            all_cells_xr = self.cell_type.get_features_xr(f, coords_list).isel(Frame=self.current_frames)
            # old_cells = None

            for frame in all_cells_xr.coords['Frame'].values:
                sys.stdout.write(f'\rGetting tracklets: Frame {frame + 1}/{f[self.masks_ds].shape[0]}')
                sys.stdout.flush()
                current_cells = all_cells_xr.isel(Frame=frame).load()
                current_cells = xr.concat([current_cells[dim] for dim in coords_list], dim='Feature')
                # Add features and cell index coordinates
                current_cells = current_cells.assign_coords({'Feature': ('Feature', coords_list),
                                                             'Cell Index': ('Cell Index', np.arange(current_cells.sizes['Cell Index']))
                                                             })
                
                # Drop nan cells (initial idx preserved in coord). Allows distances between all cells to be calculated.
                current_cells = current_cells.dropna(dim='Cell Index', how='all')
                
                if self.old_cells is not None:
                    lut, old_cells = self.frame_to_frame_matching(self.old_cells, current_cells, max_dist)
                    self.apply_lut(frame, lut, f)
                else:
                    self.old_cells = current_cells


                if frame % 100 == 0:
                    gc.collect()

                
    def frame_to_frame_matching(self, old_cells: Optional[xr.DataArray], current_cells: xr.DataArray, max_dist: int) -> tuple[dict, xr.DataArray]:
        """
        Reindex current cells to match with old_cells using linear sum assignemnt to minimise total distance between cells.
        Update mask and cells_ds with new indices.
        Idxs/cell idxs refers to the index assigned to each cell. Pos refers to the position of a given cell index in an array.
        Parameters:
            old_cells: xr.DataArray with 'idx', 'x', 'y' coords in 'Features' dim (no np.nan values)
            current_cells: xr.DataArray with 'idx', 'x', 'y' columns (no np.nan values)
        Return:
            LUT for reindexing cells
            Reindexed current cells to be used as old_cells in next frame.
        """
        current_idxs = current_cells.coords['Cell Index']
        old_idxs = old_cells.coords['Cell Index']

        distances = np.linalg.norm(
                    np.expand_dims(old_cells.sel(Feature=['X', 'Y']).values, 2) -
                    np.expand_dims(current_cells.sel(Feature=['X', 'Y']).values, 1),
                    axis=0
                )

        old_pos, current_pos = linear_sum_assignment(distances)
        valid_pairs = distances[old_pos, current_pos] < max_dist

        # Get actual cell idxs (not position idxs)
        matched_old_idxs = old_idxs[old_pos[valid_pairs]]
        matched_current_idxs = current_idxs[current_pos[valid_pairs]]
        
        unmatched_current_idxs = current_idxs[~np.isin(current_idxs, matched_current_idxs)]
        
        current_idxs = np.append(matched_current_idxs, unmatched_current_idxs)
        old_idxs = np.append(matched_old_idxs, 
                                np.arange(
                                    np.max(old_idxs)+1, 
                                    np.max(old_idxs)+len(unmatched_current_idxs)+1)
                            ).astype(int)
        
        assert len(old_idxs) == len(current_idxs), "Old idxs and new idxs different lengths" 
        # lut = np.full(np.max(current_idxs)+2, -1)
        # lut[current_idxs] = old_idxs

        lut = {int(curr): int(old) for curr, old in zip(current_idxs, old_idxs)}

        # new_current_idxs = lut[current_cells.coords['Cell Index']]

        def map_index(idx):
            return lut.get(idx)
        
        new_current_idxs = xr.apply_ufunc(
            np.vectorize(map_index),
            current_cells.coords['Cell Index'],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[int],
        ).values
        current_cells = current_cells.assign_coords({'Cell Index': ('Cell Index', new_current_idxs)})
        return lut, current_cells
        
    def apply_lut(self, frame: int, lut: dict, h5py_file: h5py.File) -> None:
        """
        In the given frame, apply the LUT to reindex cells in the segmentation mask and Cells group.
        """
        # Apply to mask
        mask = h5py_file[self.masks_ds][frame][:]
        # new_mask = np.where(mask==-1, -1, lut[mask])
        new_mask = np.vectorize(lambda idx: lut.get(idx, -1))(mask)
        h5py_file[self.masks_ds][frame] = new_mask

        valid_current_idxs = np.array(list(lut.keys()), dtype=int)   # cell indices in current frame
        valid_new_idxs = np.array(list(lut.values()), dtype=int)     # global new indices to assign

        for feature_name, feature_data in h5py_file[self.cells_group].items():
            # Get original feature data for the current frame, for mapped cell indices
            frame_data = feature_data[frame]
            feature_values = frame_data[valid_current_idxs]  # shape: (num_mapped_cells,)

            # Create reindexed array with NaNs for unmapped cells
            max_idx = int(valid_new_idxs.max()) + 1
            reindexed_feature_data = np.full(max_idx, np.nan)
            reindexed_feature_data[valid_new_idxs] = feature_values

            # Resize dataset if necessary (grow in chunks to avoid frequent resizing)
            current_shape = h5py_file[self.cells_group][feature_name].shape[1]
            if max_idx > current_shape:
                h5py_file[self.cells_group][feature_name].resize(max_idx + 100, axis=1)

            # Assign reindexed data
            h5py_file[self.cells_group][feature_name][frame, 0:max_idx] = reindexed_feature_data


    # def get_tracklet_endpoints(self, h5py_file: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """Get the time and position coordiantes of the start and end of each tracklet."""
    #     coords = self.cell_type.get_features_xr(h5py_file, ['X', 'Y']).isel(Frame=self.current_frames)
    #     coords = np.stack([coord_array.values for coord_array in coords.data_vars.values()], axis=2)

    #     not_nan = ~np.isnan(coords[:,:,0])

    #     start_frames = np.argmax(not_nan, axis=0)
    #     end_frames = len(not_nan) - 1 - np.argmax(not_nan[::-1], axis=0)

    #     start_coords = coords[start_frames, np.arange(len(start_frames))]
    #     end_coords = coords[end_frames, np.arange(len(end_frames))]

    #     return start_frames, start_coords, end_frames, end_coords

    def get_tracklet_endpoints(self, h5py_file: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the time (Frame coordinates) and position (X, Y) of the start and end of each tracklet.
        Uses the Frame coordinate values instead of just index positions.
        """
        # Get Dask-backed xarray Dataset
        coords_ds = self.cell_type.get_features_xr(h5py_file, ['X', 'Y']).isel(Frame=self.current_frames)

        # Get the actual Frame coordinate values (not just indices)
        frame_coords = coords_ds.coords['Frame'].values  # e.g., array([0, 1, 2, ..., N])

        # Compute non-NaN mask from one feature (e.g., 'X')
        valid = ~np.isnan(coords_ds['X'])

        # Compute start and end frame indices (per cell)
        start_idx = valid.argmax(dim='Frame').compute().values
        end_idx = (valid[::-1].argmax(dim='Frame')).compute().values
        end_idx = len(valid['Frame']) - 1 - end_idx  # convert reversed index

        # Map those indices to actual frame coordinate values
        start_frames = frame_coords[start_idx]
        end_frames = frame_coords[end_idx]

        # Extract the corresponding coordinates from X and Y at those frames
        x_vals = coords_ds['X'].transpose('Frame', 'Cell Index').compute().values
        y_vals = coords_ds['Y'].transpose('Frame', 'Cell Index').compute().values

        start_coords = np.stack([x_vals[start_idx, np.arange(len(start_idx))],
                                y_vals[start_idx, np.arange(len(start_idx))]], axis=1)
        
        end_coords = np.stack([x_vals[end_idx, np.arange(len(end_idx))],
                            y_vals[end_idx, np.arange(len(end_idx))]], axis=1)

        return start_frames, start_coords, end_frames, end_coords



    def join_tracklets(self, max_time_gap: int = SETTINGS.FRAME_MEMORY, max_dist: int = SETTINGS.MAXIMUM_DISTANCE_THRESHOLD) -> None:
        """Using start and end coordinates of each tracklets, match up tracklets belonging to the same cell
        """
        with h5py.File(self.file, 'r+') as f:
            all_cell_idxs = np.arange(f[self.cells_group]['X'].shape[1])

            f[self.cells_group].attrs['maximum time gap in tracks'] = max_time_gap
            
            start_frames, start_coords, end_frames, end_coords = self.get_tracklet_endpoints(f)

            dist_weights = np.linalg.norm(np.expand_dims(start_coords, 0) - np.expand_dims(end_coords, 1), axis=2)
            time_weights = np.expand_dims(start_frames, 0) - np.expand_dims(end_frames, 1)

            weights = np.where((time_weights>0) & (time_weights<max_time_gap) & (dist_weights<(max_dist*time_weights)), dist_weights, np.inf)
            potential_idxs_0, potential_idxs_1 = ~np.all(np.isinf(weights), axis=1), ~np.all(np.isinf(weights), axis=0)

            potential_dist_weights = dist_weights[potential_idxs_0][:, potential_idxs_1]
            potential_time_weights = time_weights[potential_idxs_0][:, potential_idxs_1]

            old, new = linear_sum_assignment(potential_dist_weights)
            valid_pairs = (potential_dist_weights[old, new] < max_dist) & (potential_time_weights[old, new] < max_time_gap) & (potential_time_weights[old, new] > 0)

            old, new = old[valid_pairs], new[valid_pairs]
            cells_0 = all_cell_idxs[potential_idxs_0][old]
            cells_1 = all_cell_idxs[potential_idxs_1][new]

            queue = np.column_stack((cells_0, cells_1))

            for i, (cell_0, cell_1) in enumerate(queue):
                sys.stdout.write(f'\rJoining tracklets for frames {self.current_frames.start+1} to {self.current_frames.stop+1}: Progress {int(((i + 1)/len(queue))*100):03}%')
                sys.stdout.flush()

                # Update cells group
                for feature_name, feature_data in f[self.cells_group].items():
                    cell_1_ds = feature_data[start_frames[cell_1]:, cell_1]
                    f[self.cells_group][feature_name][self.current_frames, cell_0] = cell_1_ds

                    # Empty cell_1 dataset
                    f[self.cells_group][feature_name][self.current_frames, cell_1] = np.full(feature_data.shape[0], np.nan)
                
                # Update masks
                for frame in range(start_frames[cell_1], end_frames[cell_1]+1):
                    mask = f[self.masks_ds][frame][:]
                    mask[mask==cell_1] = cell_0
                    f[self.masks_ds][frame] = mask

                # Update queue
                queue[:, 0][queue[:, 0]==cell_1] = cell_0

    def remove_short_tracks(self, min_track_length: int = SETTINGS.MINIMUM_TRACK_LENGTH) -> None:
        """Remove tracks shorter than min_track_length"""
        with h5py.File(self.file, 'r+') as f:
            
            f[self.cells_group].attrs['minimum track length'] = min_track_length
            start_frames, _, end_frames, _ = self.get_tracklet_endpoints(f)
            track_lengths = end_frames - start_frames
            short_tracks = np.nonzero(track_lengths < min_track_length)[0]

            # Fill short tracks with np.nan
            for feature_name, feature_data in f[self.cells_group].items():
                feature_array = feature_data[...]
                feature_array[:, short_tracks] = np.nan
                f[self.cells_group][feature_name][...] = feature_array

            # Remove masks for these cells
            for frame in range(f[self.masks_ds].shape[0]):
                mask = f[self.masks_ds][frame][:]
                mask[np.isin(mask, short_tracks)] = -1
                f[self.masks_ds][frame] = mask

    def delete_nan_cells(self) -> None:
        """For any cells which are only np.nan values, remove and reindex remaining cells."""
        with h5py.File(self.file, 'r+') as f:
            # Get coords to determine np.nan cells
            coords = self.cell_type.get_features_xr(f, ['X', 'Y']).isel(Frame=self.current_frames)
            coords = np.stack([coord_array.values for coord_array in coords.data_vars.values()], axis=2)
            
            old_cell_idxs = np.nonzero(np.any(~np.isnan(coords), axis=(0,2)))[0]
            new_cell_idxs = np.arange(len(old_cell_idxs))

            lut = {int(old): int(new) for old, new in zip(old_cell_idxs, new_cell_idxs)}

            if len(lut) > 0:
                for frame in range(coords.coords['Frame'].values):
                    sys.stdout.write(f'\rDeleting np.nan cells: Frame {frame+1} / {coords.shape[0]}')
                    sys.stdout.flush()
                    self.apply_lut(frame, lut, f)
                
                self.resize_datasets(f)

                # for feature_name in f[self.cells_group].keys():
                    
                #     f[self.cells_group][feature_name].resize(len(new_cell_idxs), axis=1)

    def resize_datasets(self, f: h5py.File) -> None:
        """Shrink the 'Cell Index" dimension of the cells datasets to remove nan cells."""
        sample_xarray = self.cell_type.get_features_xr(f, ['X'])
        not_nan = ~np.isnan(sample_xarray)
        not_nan_cells = not_nan.any(dim='Frame')
        not_nan_cells = np.nonzero(not_nan_cells.values)[0]

        last_not_nan_cell = int(not_nan_cells[-1])

        for feature_name in f[self.cells_group].keys():
            f[self.cells_group][feature_name].resize(last_not_nan_cell+1, axis=1)
        # # Repack to clear unused storage
        # tools.repack_hdf5(self.file)

def main():
    my_tracker = Tracker()
    my_tracker.track()
    # my_tracker.get_cell_info()
    # my_tracker.get_tracklets()
    # my_tracker.join_tracklets()
    # my_tracker.remove_short_tracks()
    # my_tracker.delete_nan_cells()
    
if __name__ == '__main__':
    main()