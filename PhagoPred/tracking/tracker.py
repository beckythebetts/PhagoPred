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

import numpy as np
import xarray as xr
import h5py
import torch
import gc
import sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple

class Tracker:

    def __init__(self, file: Path = SETTINGS.DATASET, channel: str = 'Phase', frame_batch_size: int = 10) -> None:
        if file is not None:
            self.file = file
            self.channel = channel
            self.frame_batch_size = frame_batch_size

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = extract_features.FeaturesExtraction(self.file)
            self.cell_type = self.feature_extractor.get_cell_type(self.channel)
            self.masks_ds = f'Segmentations/{self.channel}'
            self.cells_group = f'Cells/{self.channel}'

            self.global_mapping = {}

    def track(self):
        self.get_cell_info()
        self.get_tracklets()
        self.apply_lut()
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
            all_cells_xr = self.cell_type.get_features_xr(f, coords_list)
            self.old_cells = None

            for frame in all_cells_xr.coords['Frame'].values:
                sys.stdout.write(f'\rGetting tracklets: Frame {frame + 1}/{f[self.masks_ds].shape[0]}')
                sys.stdout.flush()
                current_cells = all_cells_xr.sel(Frame=frame).load()
                current_cells = xr.concat([current_cells[dim] for dim in coords_list], dim='Feature')
                # Add features and cell index coordinates
                current_cells = current_cells.assign_coords({'Feature': ('Feature', coords_list),
                                                             'Cell Index': ('Cell Index', np.arange(current_cells.sizes['Cell Index']))
                                                             })
                
                # Drop nan cells (initial idx preserved in coord). Allows distances between all cells to be calculated.
                current_cells = current_cells.dropna(dim='Cell Index', how='all')
                
                if self.old_cells is not None:
                    self.global_mapping[frame], self.old_cells = self.frame_to_frame_matching(self.old_cells, current_cells, max_dist)
                    # self.apply_lut(frame, lut, f)
                else:
                    self.old_cells = current_cells
       
    def frame_to_frame_matching(self, old_cells: xr.DataArray, current_cells: xr.DataArray, max_dist: int) -> tuple[dict, xr.DataArray]:
        """
        Match current_cells to old_cells minimizing centroid distance,
        assign unmatched cells new global IDs.

        Returns:
            lut: dict mapping current_cells local IDs -> global track IDs
            current_cells with updated 'Cell Index' coords (global IDs)
        """

        current_idxs = current_cells.coords['Cell Index'].values
        old_idxs = old_cells.coords['Cell Index'].values  # global IDs from previous frame

        # Calculate distances (shape: old_cells x current_cells)
        distances = np.linalg.norm(
            old_cells.sel(Feature=['X', 'Y']).values[:, :, None] - 
            current_cells.sel(Feature=['X', 'Y']).values[:, None, :], axis=0
        )

        old_pos, current_pos = linear_sum_assignment(distances)

        # Filter matches within max_dist
        valid_pairs_mask = distances[old_pos, current_pos] < max_dist
        matched_old_idxs = old_idxs[old_pos[valid_pairs_mask]]
        matched_current_idxs = current_idxs[current_pos[valid_pairs_mask]]

        # Unmatched current cells (new tracks)
        unmatched_current_mask = ~np.isin(current_idxs, matched_current_idxs)
        unmatched_current_idxs = current_idxs[unmatched_current_mask]

        self.max_global_id = old_idxs.max() if len(old_idxs) > 0 else -1
        new_global_ids = np.arange(self.max_global_id + 1, self.max_global_id + 1 + len(unmatched_current_idxs))

        # Build LUT: current frame local ID -> global track ID
        lut = {int(curr): int(old) for curr, old in zip(matched_current_idxs, matched_old_idxs)}
        lut.update({int(curr): int(new) for curr, new in zip(unmatched_current_idxs, new_global_ids)})

        # Update current_cells with global track IDs
        new_current_idxs = np.array([lut[idx] for idx in current_idxs])
        current_cells = current_cells.assign_coords({'Cell Index': ('Cell Index', new_current_idxs)})

        return lut, current_cells

        
    def apply_lut(self):
        with h5py.File(self.file, 'r+') as h5py_file:
            print('\nUpdating dataset with tracklets...')

            masks = h5py_file[self.masks_ds][:]
            new_masks = np.full_like(masks, -1)

            features = {name: h5py_file[self.cells_group][name] for name in h5py_file[self.cells_group].keys()}

            # Find max global index needed for resizing
            max_new_idx = 0
            for lut in self.global_mapping.values():
                if lut:
                    max_new_idx = max(max_new_idx, max(lut.values()))

            # Resize feature datasets once with some buffer
            for feature_name, feature_data in features.items():
                curr_size = feature_data.shape[1]
                if max_new_idx >= curr_size:
                    new_size = max_new_idx + 1
                    feature_data.resize((feature_data.shape[0], new_size))
                else:
                    new_size = feature_data.shape[1]

            # Build LUT arrays for fast mapping per frame
            lut_arrays = {}
            for frame, lut in self.global_mapping.items():
                if not lut:
                    continue
                max_old_idx = max(lut.keys())
                lut_arr = np.full(max_old_idx + 2, -1, dtype=int)
                for old_idx, new_idx in lut.items():
                    lut_arr[old_idx] = new_idx
                lut_arrays[frame] = lut_arr

            # Apply LUTs to masks and features per frame
            for i, frame in enumerate(range(h5py_file[self.masks_ds].shape[0])):
                lut_arr = lut_arrays.get(frame, None)
                if lut_arr is None:
                    # No LUT, copy original mask
                    new_masks[i] = masks[i]
                    continue

                # Map mask labels using LUT array
                mask = masks[i]
                clipped_mask = np.clip(mask, 0, len(lut_arr) - 1)
                new_masks[i] = lut_arr[clipped_mask]

                # Reindex features using LUT dict
                lut = self.global_mapping[frame]
                valid_old_idxs = np.array(list(lut.keys()), dtype=int)
                valid_new_idxs = np.array(list(lut.values()), dtype=int)

                for feature_name, feature_data in features.items():
                    frame_feature = feature_data[frame]
                    feature_values = frame_feature[valid_old_idxs]

                    new_frame_feature = np.full(new_size, np.nan)
                    # Assign values to new global indices
                    new_frame_feature[valid_new_idxs] = feature_values
                    feature_data[frame] = new_frame_feature

            # Write updated masks back to file
            h5py_file[self.masks_ds][:] = new_masks

    def get_tracklet_endpoints(self, h5py_file: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the time (Frame coordinates) and position (X, Y) of the start and end of each tracklet.
        Uses the Frame coordinate values instead of just index positions.
        """
        # Get Dask-backed xarray Dataset
        coords_ds = self.cell_type.get_features_xr(h5py_file, ['X', 'Y'])

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
                
                sys.stdout.write(f'\rJoining tracklets: Progress {int(((i + 1)/len(queue))*100):03}%')
                sys.stdout.flush()

                # Update cells group
                for feature_name, feature_data in f[self.cells_group].items():
                    cell_1_ds = feature_data[start_frames[cell_1]:, cell_1]
                    f[self.cells_group][feature_name][start_frames[cell_1]:, cell_0] = cell_1_ds

                    cell_1_shape = f[self.cells_group][feature_name][:, cell_1].shape
                    # Empty cell_1 dataset
                    f[self.cells_group][feature_name][:, cell_1] = np.full(cell_1_shape, np.nan)
                
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
            print('\nRemoving emty cells...')
            # Get coords to determine np.nan cells
            coords = self.cell_type.get_features_xr(f, ['X', 'Y'])
            coords_np = np.stack([coord_array.values for coord_array in coords.data_vars.values()], axis=2)
            
            old_cell_idxs = np.nonzero(np.any(~np.isnan(coords_np), axis=(0,2)))[0]
            new_cell_idxs = np.arange(len(old_cell_idxs))

            lut = {int(old): int(new) for old, new in zip(old_cell_idxs, new_cell_idxs)}
            
            frames = coords.coords['Frame'].values
            self.global_mapping = {frame: lut for frame in frames}
            self.apply_lut()
            self.resize_datasets(f)

    def resize_datasets(self, f: h5py.File) -> None:
        """Shrink the 'Cell Index" dimension of the cells datasets to remove nan cells."""
        print('\nResizing Dataset...')
        # sample_array = self.cell_type.get_features_xr(f, ['X'])['X']
        # not_nan = ~np.isnan(sample_array)
        # # not_nan_cells = not_nan.any(dim='Frame')
        # not_nan_cells = np.any(not_nan, axis=0)
        # not_nan_cells = np.nonzero(not_nan_cells.values)[0]

        # last_not_nan_cell = int(not_nan_cells[-1])
        # print(last_not_nan_cell)

        for feature_name in f[self.cells_group].keys():
            # f[self.cells_group][feature_name].resize(last_not_nan_cell+1, axis=1)
            f[self.cells_group][feature_name].resize(self.max_global_id+1, axis=1)
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