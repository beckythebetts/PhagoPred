import h5py
import numpy as np
import trackpy as tp
from tqdm import tqdm
import pandas as pd

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features

# tp.linking.Linker.MAX_SUB_NET_SIZE = 35 # Only include for datasets with many cells
# tp.link()

class TrackerWithTrackpy:
    def __init__(self, file_path, channel='Phase'):
        self.file = file_path
        self.channel = channel
        self.masks_ds = f'Segmentations/{self.channel}'
        self.cells_group = f'Cells/{self.channel}'
        self.max_track_id = 0
        self.track_id_map = None  # will be a 2D array: frames x max_cells
        self.min_cell_size = 500

        self.feature_extractor = extract_features.FeaturesExtraction(self.file)

    def extract_features_for_tracking(self):
        # Load coordinates (X, Y) for all cells in all frames
        coords_list = ['X', 'Y']
        with h5py.File(self.file, 'r') as f:
            coords = []
            num_frames = f[self.masks_ds].shape[0]
            for frame in tqdm(range(num_frames), desc='Gathering cell coords: '):
                # Load cell coordinates for current frame
                X = f[f'{self.cells_group}/X'][frame]
                Y = f[f'{self.cells_group}/Y'][frame]
                areas = f[f'{self.cells_group}/Area'][frame]

                valid_idxs = np.nonzero(~np.isnan(X) & ~np.isnan(Y) & (areas > self.min_cell_size))[0]

                for idx in valid_idxs:
                    coords.append({'frame': frame, 'x': X[idx], 'y': Y[idx], 'cell_index': idx})

        return coords, num_frames

    def track(self, search_range=10, memory=3, scale_coords=1):
        """
        Run tracking with trackpy and store mapping of cell indices to track IDs
        scale_coords, allows larger search_range iwhtout rquiring more memoery
        """
        coords, num_frames = self.extract_features_for_tracking()
        search_range /= scale_coords  # Adjust search range based on coordinate scaling

        # tp.linking.utils.MAX_SUB_NET_SIZE = 50
        tp.linking.Linker.MAX_SUB_NET_SIZE = 50
        # tp.link()
        # Run trackpy linking
        coords = pd.DataFrame(coords)
        coords['x'] /= scale_coords
        coords['y'] /= scale_coords
        df = tp.link_df(pd.DataFrame(coords), search_range=search_range, memory=memory)

        df['x'] *= scale_coords
        df['y'] *= scale_coords
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
            
        with h5py.File(self.file, 'r+') as f:
            f['Segmentations']['Phase'].attrs['Tracking search range'] = search_range
            f['Segmentations']['Phase'].attrs['Tracking memory'] = memory

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

    # def remove_nan_tracks(self, max_nan_amount: float = 0.3) -> None:
    #     """Remove tracks where greater than max_nan_amount proprortion of frames are nan valued (missing)."""
    #     self.feature_extractor = extract_features.FeaturesExtraction(self.file)
        
    #     self.feature_extractor.add_feature(features.FirstLastFrame(), cell_type=self.channel)
    #     self.feature_extractor.set_up()
    #     self.feature_extractor.extract_features()
        
    
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
        # found_features = True
        # with h5py.File(self.file, 'r') as f:
        #     for feature_name in features.Coords().get_names():
        #         if feature_name not in f[self.cells_group].keys():
        #             found_features = False
        # if not found_features:
        self.feature_extractor.add_feature(features.Coords(), cell_type=self.channel)
        self.feature_extractor.set_up()
        self.feature_extractor.extract_features(crop=False)
        # else:
        #     print('Features already found, skipping to tracking.')
            
        
def run_tracking(dataset=SETTINGS.DATASET):
    tracker = TrackerWithTrackpy(dataset, 'Phase')
    
    tracker.get_cell_info()
    tracker.track(search_range=SETTINGS.MAXIMUM_DISTANCE_THRESHOLD, memory=SETTINGS.FRAME_MEMORY)  # tune these params as needed
    tracker.apply_track_id_map()
    
def main():
    run_tracking()

if __name__ == "__main__":
    main()

