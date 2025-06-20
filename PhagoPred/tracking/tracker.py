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
        with h5py.File(self.file, 'r+') as f:
            # self.masks_ds = f['Segmentations'][self.channel]
            # self.cells_group = f['Cells'][self.channel]
            
            self.cells_ds.attrs['minimum distance'] = min_dist

            all_cells_xr = self.cell_type.get_features_xr(f, ['X', 'Y'])
            old_cells = None
            for frame in range(f[self.masks_ds].shape[0]):
                current_cells = all_cells_xr.isel(Frame=frame).sel(Feature=('X', 'Y'))
                # Add 'idx' feature and drop np.nan cells
                current_cells = current_cells.assign_coords(
                    idx=('Feature', np.arange(current_cells.size['Cell Index'])))
                current_cells = current_cells.dropna(dim='Cell Index', how='all')

                lut, old_cells = self.frame_to_frame_matching(old_cells, current_cells, min_dist)
                self.apply_lut(frame, lut, f)
                old_cells = current_cells
                # Apply lut to segmentation mask and reorder Cells dataset
                
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
        # need no np.nan values in reindexed current cells (need to add idx column to xr.DataArray)
        if old_cells is not None:

            current_idxs = current_cells.sel(Feature='idx').values.astype(int)
            old_idxs = old_cells.sel(Feature='idx').values.astype(int)

            distances = np.linalg.norm(
                        np.expand_dims(old_cells.sel(Feature=['x', 'y']).values, 1) -
                        np.expand_dims(current_cells.sel(Feature=['x', 'y']).values, 0),
                        axis=2
                    )

            old_pos, current_pos = linear_sum_assignment(distances)
            valid_pairs = distances[old_pos, current_pos] < min_dist
            # get actual cell idxs (not position idxs)
            matched_old_idxs = old_cells.sel(Feature='idx').values[old_pos[valid_pairs]]
            matched_current_idxs = current_cells.sel(Feature='idx').values[current_pos[valid_pairs]]
            
            # if len(current_cells) > len(current_idxs):
                # add missing current_idxs to end of current_idxs, and add idxs to old_idxs. Current_idxs will be transformed to old_idxs.
                
            # unmatched_current_idxs = current_idxs[~current_idxs.isin(matched_current_idxs)]
            unmatched_current_idxs = current_idxs[~np.isin(current_idxs, matched_current_idxs)]
            
            current_idxs = np.append(matched_current_idxs, unmatched_current_idxs)
            old_idxs = np.append(matched_old_idxs, 
                                    np.arange(
                                        np.max(old_cells.sel(Feature='idx').values)+1, 
                                        np.max(old_cells.sel(Feature='idx'))+len(unmatched_current_idxs)+1)
                                ).astype(int)
            
            assert len(old_idxs) == len(current_idxs), "Old idxs and new idxs different lengths" 
            #update mask
            lut = np.zeros(np.max(current_idxs)+1)
            lut[current_idxs] = old_idxs

            current_cells.loc[dict(Feature='idx')] = lut[current_cells.sel(Feature='idx')]

            return lut, current_cells
        
    def apply_lut(self, frame: int, lut: np.ndarray, h5py_file: h5py.File) -> None:
        """
        In the given frame, apply the LUT to reindex cells in the segmentation mask and Cells group.
        """
        h5py_file[self.masks_ds][frame] = lut[self.masks_ds[frame][:]]
        for feature_name, feature_data in h5py_file[self.cells_group].items():
            reindexed_feature_data = np.full(np.max(lut)+1, np.nan)
            reindexed_feature_data[lut] = feature_data[()]
            h5py_file[self.cells_group][feature_name] = reindexed_feature_data
            # h5py_file[self.cells_group][feature_name] = h5py_file[self.cells_group][feature_name][lut]

def main():
    my_tracker = Tracker()
    my_tracker.get_cell_info()
    my_tracker.get_tracklets()
    
if __name__ == '__main__':
    main()