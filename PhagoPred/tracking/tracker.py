import sys

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

    def __init__(self, file: Path = SETTINGS.DATASET, channel: str = 'Phase') -> None:
        if file is not None:
            self.file = file
            self.channel = channel
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = extract_features.FeaturesExtraction(self.file)
            self.cell_type = self.feature_extractor.get_cell_type(self.channel)
            self.masks_ds = f'Segmentations/{self.channel}'
            self.cells_group = f'Cells/{self.channel}'

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
            old_cells = None

            for frame in range(f[self.masks_ds].shape[0]):
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
                
                if old_cells is not None:
                    lut, old_cells = self.frame_to_frame_matching(old_cells, current_cells, max_dist)
                    self.apply_lut(frame, lut, f)
                else:
                    old_cells = current_cells

                
    def frame_to_frame_matching(self, old_cells: Optional[xr.DataArray], current_cells: xr.DataArray, max_dist: int) -> tuple[np.array, xr.DataArray]:
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
        lut = np.full(np.max(current_idxs)+2, -1)
        lut[current_idxs] = old_idxs

        new_current_idxs = lut[current_cells.coords['Cell Index']]
        current_cells = current_cells.assign_coords({'Cell Index': ('Cell Index', new_current_idxs)})
        return lut.astype(int), current_cells
        
    def apply_lut(self, frame: int, lut: np.ndarray, h5py_file: h5py.File) -> None:
        """
        In the given frame, apply the LUT to reindex cells in the segmentation mask and Cells group.
        """
        # Apply to mask
        mask = h5py_file[self.masks_ds][frame][:]
        new_mask = np.where(mask==-1, -1, lut[mask])
        h5py_file[self.masks_ds][frame] = new_mask

        valid_mask = np.nonzero(lut>=0)[0]
        valid_lut = lut[valid_mask]
        # Apply to all cell features
        for feature_name, feature_data in h5py_file[self.cells_group].items():
            feature_data = feature_data[frame][valid_mask]
            reindexed_feature_data = np.full(np.max(valid_lut.astype(int))+1, np.nan)
            reindexed_feature_data[valid_lut] = feature_data
            # Resize dataset if necassary
            if len(reindexed_feature_data) > h5py_file[self.cells_group][feature_name].shape[1]:
                h5py_file[self.cells_group][feature_name].resize(len(reindexed_feature_data), axis=1)
            h5py_file[self.cells_group][feature_name][frame, 0:len(reindexed_feature_data)] = reindexed_feature_data


    def get_tracklet_endpoints(self, h5py_file: h5py.File) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the time and position coordiantes of the start and end of each tracklet."""
        coords = self.cell_type.get_features_xr(h5py_file, ['X', 'Y'])
        coords = np.stack([coord_array.values for coord_array in coords.data_vars.values()], axis=2)

        not_nan = ~np.isnan(coords[:,:,0])

        start_frames = np.argmax(not_nan, axis=0)
        end_frames = len(not_nan) - 1 - np.argmax(not_nan[::-1], axis=0)

        start_coords = coords[start_frames, np.arange(len(start_frames))]
        end_coords = coords[end_frames, np.arange(len(end_frames))]

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
                    f[self.cells_group][feature_name][-len(cell_1_ds):, cell_0] = cell_1_ds

                    # Empty cell_1 dataset
                    f[self.cells_group][feature_name][:, cell_1] = np.full(feature_data.shape[0], np.nan)
                
                # Update masks
                for frame in range(start_frames[cell_1], end_frames[cell_1]+1):
                    mask = f[self.masks_ds][frame][:]
                    mask[mask==cell_1] = cell_0
                    f[self.masks_ds][frame][...] = mask

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

    def delete_nan_cells(self):
        """For any cells which are only np.nan values, remove and reindex remaining cells."""
        with h5py.File(self.file, 'r+') as f:
            # Get coords to determine np.nan cells
            coords = self.cell_type.get_features_xr(f, ['X', 'Y'])
            coords = np.stack([coord_array.values for coord_array in coords.data_vars.values()], axis=2)

            old_cell_idxs = np.nonzero(np.any(~np.isnan(coords), axis=(0,2)))[0]
            new_cell_idxs = np.arange(len(old_cell_idxs))
            # lut = np.zeros(coords.shape[1])
            lut = np.full(coords.shape[1], -1)
            lut[old_cell_idxs] = new_cell_idxs
            for frame in range(coords.shape[0]):
                sys.stdout.write(f'\rDeleting np.nan cells: Frame {frame+1} / {coords.shape[0]}')
                sys.stdout.flush()
                self.apply_lut(frame, lut.astype(int), f)

            for feature_name in f[self.cells_group].keys():
                f[self.cells_group][feature_name].resize(len(new_cell_idxs), axis=1)

        # # Repack to clear unused storage
        # tools.repack_hdf5(self.file)

def main():
    my_tracker = Tracker()
    # my_tracker.get_cell_info()
    # my_tracker.get_tracklets()
    # my_tracker.join_tracklets()
    # my_tracker.remove_short_tracks()
    my_tracker.delete_nan_cells()
    
if __name__ == '__main__':
    main()