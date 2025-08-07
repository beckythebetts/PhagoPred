import h5py
import numpy as np
import trackpy as tp
from tqdm import tqdm
import pandas as pd

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features

class TrackerWithTrackpy:
    def __init__(self, file_path, channel='Phase'):
        self.file = file_path
        self.channel = channel
        self.masks_ds = f'Segmentations/{self.channel}'
        self.cells_group = f'Cells/{self.channel}'
        self.max_track_id = 0
        self.track_id_map = None  # will be a 2D array: frames x max_cells

        self.feature_extractor = extract_features.FeaturesExtraction(self.file)

    def extract_features_for_tracking(self):
        # Load coordinates (X, Y) for all cells in all frames
        coords_list = ['X', 'Y']
        with h5py.File(self.file, 'r') as f:
            coords = []
            num_frames = f[self.masks_ds].shape[0]
            for frame in range(num_frames):
                # Load cell coordinates for current frame
                X = f[f'{self.cells_group}/X'][frame]
                Y = f[f'{self.cells_group}/Y'][frame]
                # Filter out NaNs or invalid points (assuming nan means missing)
                valid = ~np.isnan(X) & ~np.isnan(Y)
                frame_coords = np.vstack((X[valid], Y[valid])).T
                for idx, (x, y) in enumerate(frame_coords):
                    coords.append({'frame': frame, 'x': x, 'y': y, 'cell_index': idx})
        return coords, num_frames

    def track(self, search_range=10, memory=3):
        """
        Run tracking with trackpy and store mapping of cell indices to track IDs
        """
        coords, num_frames = self.extract_features_for_tracking()

        # Run trackpy linking
        df = tp.link_df(pd.DataFrame(coords), search_range=search_range, memory=memory)

        # Find max track id assigned
        self.max_track_id = df['particle'].max()

        # Prepare track_id_map: shape (num_frames, max_cells)
        # We'll assign -1 if no tracking info
        max_cells = max(df['cell_index'].max() + 1, 1)  # cell indices are local per frame
        self.track_id_map = -np.ones((num_frames, max_cells), dtype=int)

        # Populate track_id_map
        for row in df.itertuples():
            # row.frame, row.cell_index, row.particle
            self.track_id_map[row.frame, row.cell_index] = row.particle

    def apply_track_id_map(self):
        with h5py.File(self.file, 'r+') as f:

            self.remove_short_tracks()
            print('\nUpdating datasets with track ID map...\n')
            # Resize cells datasets if necessary
            for feature_data in f[self.cells_group].values():
                if self.max_track_id + 1 > feature_data.shape[1]:
                    feature_data.resize(self.max_track_id + 1, axis=1)

            for frame, track_id_map in enumerate(tqdm(self.track_id_map)):
                mask = f[self.masks_ds][frame][:]
                # Replace mask indices with track IDs, keep -1 for background
                f[self.masks_ds][frame] = np.where(mask == -1, -1, track_id_map[mask])

                valid_cell_idxs = np.nonzero(track_id_map >= 0)[0]
                valid_track_id_map = track_id_map[valid_cell_idxs]

                for feature_name, feature_data in f[self.cells_group].items():
                    feature_vals = feature_data[frame][valid_cell_idxs]
                    reindexed_feature_data = np.full(self.max_track_id + 1, np.nan)
                    reindexed_feature_data[valid_track_id_map] = feature_vals
                    f[self.cells_group][feature_name][frame, :len(reindexed_feature_data)] = reindexed_feature_data

            for feature_data in f[self.cells_group].values():
                feature_data.resize(self.max_track_id + 1, axis=1)

    def remove_short_tracks(self, min_track_length: int = SETTINGS.MINIMUM_TRACK_LENGTH) -> None:
        """Remove any tracks with less than {min_track_length} instances"""
        track_ids, counts = np.unique(self.track_id_map[self.track_id_map >= 0], return_counts=True)
        short_track_ids = track_ids[counts <= min_track_length]

        mask = np.isin(self.track_id_map, short_track_ids)
        self.track_id_map[mask] = -1
        self.remap_track_id_map()

    
    def remap_track_id_map(self):
        """Reindex the track_id_map so all track_ids are consecutive"""
        valid_mask = self.track_id_map >= 0
        track_ids = np.unique(self.track_id_map[valid_mask])
        new_track_ids = np.arange(len(track_ids))

        id_map = -np.ones(track_ids.max()+1, dtype=int)
        id_map[track_ids] = new_track_ids

        self.track_id_map = np.where(valid_mask, id_map[self.track_id_map], -1)

        self.max_track_id = np.max(self.track_id_map)

    def get_cell_info(self):
        """
        Add each cells centroid cooridnates and areas to cells dataset
        """
        self.feature_extractor.add_feature(features.Coords(), cell_type=self.channel)
        self.feature_extractor.set_up()
        self.feature_extractor.extract_features()

def main():
    tracker = TrackerWithTrackpy(SETTINGS.DATASET, 'Phase')
    tracker.get_cell_info()
    tracker.track(search_range=SETTINGS.MAXIMUM_DISTANCE_THRESHOLD, memory=SETTINGS.FRAME_MEMORY)  # tune these params as needed
    tracker.apply_track_id_map()

if __name__ == "__main__":
    main()
