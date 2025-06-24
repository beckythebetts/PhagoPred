import sys

from typing import Optional
import torch
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
import xarray as xr

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features

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

    def get_tracklets(self, min_dist: int = SETTINGS.MINIMUM_DISTANCE_THRESHOLD) -> None:
        """Between each pair of consecutive frames, reassign the cell indexes of the second frame to minimise the total distance gap between all cells centroids.
        To join two cells, their distance gap must be less than min_dist
        """
        print('\nGetting tracklets...')
        with h5py.File(self.file, 'r+') as f:
            f[self.cells_group].attrs['minimum distance'] = min_dist
            
            coords_list = ['X', 'Y']
            all_cells_xr = self.cell_type.get_features_xr(f, coords_list)
            old_cells = None

            for frame in range(f[self.masks_ds].shape[0]):
                sys.stdout.write(f'\rFrame {frame + 1}/{f[self.masks_ds].shape[0]}')
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
                    lut, old_cells = self.frame_to_frame_matching(old_cells, current_cells, min_dist)
                    self.apply_lut(frame, lut, f)
                else:
                    old_cells = current_cells

                
    def frame_to_frame_matching(self, old_cells: Optional[xr.DataArray], current_cells: xr.DataArray, min_dist: int) -> tuple[np.array, xr.DataArray]:
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
        valid_pairs = distances[old_pos, current_pos] < min_dist
        # get actual cell idxs (not position idxs)
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
            print(valid_lut, feature_data)
            reindexed_feature_data[valid_lut] = feature_data
            # Resize dataset if necassary
            if len(reindexed_feature_data) > h5py_file[self.cells_group][feature_name].shape[1]:
                h5py_file[self.cells_group][feature_name].resize(len(reindexed_feature_data), axis=1)
            h5py_file[self.cells_group][feature_name][frame, 0:len(reindexed_feature_data)] = reindexed_feature_data

def main():
    my_tracker = Tracker()
    my_tracker.get_cell_info()
    my_tracker.get_tracklets()
    
if __name__ == '__main__':
    main()