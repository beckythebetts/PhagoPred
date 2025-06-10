from typing import Optional
import torch
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
import xarray as xr

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features

class Tracking:

    def __init__(self, file: Path = SETTINGS.DATSET, channel: str = 'Phase') -> None:
        self.file = file
        self.channel = channel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = extract_features.FeaturesExtraction(self.file)
        self.cell_type = self.feature_extractor.get_cell_type(self.channel)

    def get_cell_info(self):
        """
        Add each cells centroid cooridnates and areas to cells dataset
        """
        self.feature_extractor.add_feature(features.Coords())
        self.feature_extractor.set_up()
        self.feature_extractor.extract_features()

    def get_tracklets(self, min_dist: int = SETTINGS.MINIMUM_DISTANCE_THRESHOLD) -> None:
        with h5py.File(self.file, 'r+') as f:
            masks_ds = f['Segmentations'][self.channel]
            cells_ds = f['Cells'][self.channel]
            
            cells_ds.attrs['minimum distance'] = min_dist

            all_cells_xr = self.cell_type.get_features_xr(self.file)
            old_cells = None
            for frame in range(masks_ds.shape[0]):
                current_cells = all_cells_xr.isel(Frame=frame).sel(Feature=('X', 'Y'))
                # Add 'idx' feature and drop np.nan cells
                current_cells = current_cells.assign_coords(
                    idx=('Features', np.arange(current_cells.size['Cell Index'])))
                current_cells = current_cells.dropna(dim='Cell Index', how='all')

                old_cells = self.frame_to_frame_matching(old_cells, current_cells, min_dist)
                
    def frame_to_frame_matching(self, old_cells: Optional[xr.DataArray], current_cells: xr.DataArray, min_dist: int) -> xr.DataArray:
        """
        Reindex current cells to match with old_cells using linear sum assignemnt to minimise total distance between cells.
        Update mask and cells_ds with new indices.
        Parameters:
            old_cells: xr.DataArray with 'idx', 'x', 'y' coords in 'Features' dim (no np.nan values)
            current_cells: xr.DataArray with 'idx', 'x', 'y' columns (no np.nan values)
        Return:
            Reindexed current cells to be used as old_cells in next frame.
        """
        # need no np.nan values in reindexed current cells (need to add idx column to xr.DataArray)
        if old_cells is not None:
            distances = np.linalg.norm(
                        np.expand_dims(old_cells.values, 1) -
                        np.expand_dims(current_cells, 0),
                        axis=2
                    )

            old_idxs, current_idxs = linear_sum_assignment(distances)
            valid_pairs = distances[old_idxs, current_idxs] < min_dist
            # get actual cell idxs (not position idxs)
            old_idxs = old_cells.sel(Feature='idx').values[old_idxs[valid_pairs]]
            matched_current_idxs = current_cells.sel(Features='idx').values[current_idxs[valid_pairs]]
            
            if len(current_cells) > current_idxs:
                # add missing current_idxs to end of current_idxs, and add idxs to old_idxs. Current_idxs will be transformed to old_idxs.
                current_idxs = current_cells.sel(Features='idx').values
                unmatched_current_idxs = current_idxs[~current_idxs.isin(matched_current_idxs)]
                current_idxs = np.append(matched_current_idxs, unmatched_current_idxs)

                old_idxs = np.append(old_idxs, 
                                     np.arange(
                                         np.max(old_cells.sel(Features='idx'), 
                                         np.max(old_cells.sel(Features='idx'))+len(unmatched_current_idxs))+1)
                                    ).astype(int)
            
            #update mask
            lut = np.zeros(np.max(current_idxs)+1)
            lut[current_idxs] = old_idxs


