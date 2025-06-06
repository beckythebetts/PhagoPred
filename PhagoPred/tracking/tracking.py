import torch
from pathlib import Path
import h5py
import numpy as np

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

    def get_tracklets(self, min_dist: str = SETTINGS.MINIMUM_DISTANCE_THRESHOLD):
        with h5py.File(self.file, 'r+') as f:
            masks_ds = f['Segmentations'][self.channel]
            cells_ds = f['Segmentations'][self.channel]
            
            cells_ds.attrs['minimum distance'] = min_dist

            all_cells_xr = self.cell_type.get_features_xr(self.file)
            for frame in range(masks_ds.shape[0]):
                current_cells = all_cells_xr.isel(Frame=frame).sel(Feature=('X', 'Y'))
                if frame != 0:
                    distances = np.linalg.norm(
                        np.expand_dims(old_cells.)
                    )
