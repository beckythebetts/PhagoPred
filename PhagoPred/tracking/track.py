import sys
import gc
from typing import Optional, Tuple
from collections import defaultdict


from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import xarray as xr
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features
from PhagoPred.utils import tools

class Tracker:
    """
    Attributes
    ----------
    track_id_mapping: list of dicts mapping a cell_idx to a gloal track id for each frame
    """
    def __init__(self, file: Path = SETTINGS.DATASET, channel: str = 'Phase') -> None:
        self.file = file
        self.channel = channel

        self.feature_extractor = extract_features.FeaturesExtraction(self.file)
        self.cell_type = self.feature_extractor.get_cell_type(self.channel)
        self.masks_ds = f'Segmentations/{self.channel}'
        self.cells_group = f'Cells/{self.channel}'

        self.track_id_map = []

    def track(self):
        self.get_cell_info()
        self.get_tracklets()
        # self.join_tracklets()
        self.join_tracklets_batches()
        self.remove_short_tracks()
        self.apply_track_id_map()

    def get_cell_info(self):
        """
        Add each cells centroid cooridnates and areas to cells dataset
        """
        self.feature_extractor.add_feature(features.Coords(), cell_type=self.channel)
        self.feature_extractor.set_up()
        self.feature_extractor.extract_features()

    def get_tracklets(self, max_dist: int = SETTINGS.MAXIMUM_DISTANCE_THRESHOLD) -> None:
        """Between each pair of consecutive frames, asign each cell to a globl track_id based on it's on nearest celll in previous frame.
        To join two cells, their distance gap must be less than max_dist
        """
        print('\nGetting tracklets....\n')
        with h5py.File(self.file, 'r+') as f:
            f[self.cells_group].attrs['maximum cell distance moved between frames'] = max_dist
            
            coords_list = ['X', 'Y']
            all_cells_xr = self.cell_type.get_features_xr(f, coords_list)
            self.max_num_cells = len(all_cells_xr.coords['Cell Index'])
            self.old_cells = None

            for frame in tqdm(all_cells_xr.coords['Frame'].values):
                # sys.stdout.write(f'\rFrame {frame + 1}/{f[self.masks_ds].shape[0]}')
                # sys.stdout.flush()
                current_cells = all_cells_xr.sel(Frame=frame).load()
                current_cells = xr.concat([current_cells[dim] for dim in coords_list], dim='Feature')
                # Add features and cell index coordinates
                current_cells = current_cells.assign_coords({'Feature': ('Feature', coords_list),
                                                             'Cell Index': ('Cell Index', np.arange(current_cells.sizes['Cell Index']))
                                                             })
                
                # Drop nan cells (initial idx preserved in coord). Allows distances between all cells to be calculated.
                current_cells = current_cells.dropna(dim='Cell Index', how='all')
                
                if self.old_cells is not None:
                    frame_mapping, self.old_cells = self.frame_to_frame_matching(self.old_cells, current_cells, max_dist)
                    self.track_id_map.append(frame_mapping)
                else:
                    self.old_cells = current_cells

                    # Initialise global tracks with idxs of cells in first frame
                    track_id_map = -np.ones(self.max_num_cells, dtype=int)
                    track_id_map[current_cells.coords['Cell Index'].values] = current_cells.coords['Cell Index'].values
                    self.track_id_map.append(track_id_map)

                    self.max_track_id = int(np.max(self.track_id_map[0]))
        self.track_id_map = np.stack(self.track_id_map, axis=0)
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

        
        track_id_map = -np.ones(self.max_num_cells, dtype=int)
        track_id_map[matched_current_idxs] = matched_old_idxs
        track_id_map[unmatched_current_idxs] = np.arange(self.max_track_id + 1, self.max_track_id + 1 + len(unmatched_current_idxs))

        new_current_idxs = track_id_map[current_idxs]
        current_cells = current_cells.assign_coords({'Cell Index': ('Cell Index', new_current_idxs)})

        self.max_track_id = int(max(self.max_track_id, max(track_id_map)))
        return track_id_map, current_cells
    
    def apply_track_id_map(self):
        with h5py.File(self.file, 'r+') as f:
            print('\nUpdating datasets with track ID map...\n')
            # Resize cells datasets to larger if necessary
            for feature_data in f[self.cells_group].values():
                if self.max_track_id+1 > feature_data.shape[1]:
                    feature_data.resize(self.max_track_id+1, axis=1)

            for frame, track_id_map in enumerate(tqdm(self.track_id_map)):
                # sys.stdout.write(f'\rFrame {frame + 1}/{f[self.masks_ds].shape[0]}')
                # sys.stdout.flush()
                mask = f[self.masks_ds][frame][:]
                f[self.masks_ds][frame] = np.where(mask==-1, -1, track_id_map[mask])

                valid_cell_idxs = np.nonzero(track_id_map>=0)[0]
                valid_track_id_map = track_id_map[valid_cell_idxs]

                for feature_name, feature_data in f[self.cells_group].items():
                    feature_data = feature_data[frame][valid_cell_idxs]
                    reindexed_feature_data = np.full(self.max_track_id+1, np.nan)
                    reindexed_feature_data[valid_track_id_map] = feature_data
                    f[self.cells_group][feature_name][frame, :len(reindexed_feature_data)] = reindexed_feature_data
            
            # Ensure all cells datasets are no larger than necessasry
            for feature_data in f[self.cells_group].values():
                    feature_data.resize(self.max_track_id+1, axis=1)

    def get_tracklet_endpoints(self):
        print('\nGetting tracklets endpoints...\n')
        start_frames, start_coords, end_frames, end_coords = [], [], [], []
        with h5py.File(self.file, 'r') as f:
            coords_ds = self.cell_type.get_features_xr(f, ['X', 'Y'])

            
            for track_id in tqdm(range(self.max_track_id+1)):
                # sys.stdout.write(f'\rTracklet {track_id + 1}/{self.max_track_id+1}')
                # sys.stdout.flush()
                frames, cell_ids = np.nonzero(self.track_id_map==track_id)

                start_frame, end_frame = frames[0], frames[-1]

                start_coord = np.stack((coords_ds['X'].isel({'Frame': start_frame, 'Cell Index': cell_ids[0]}),
                               coords_ds['Y'].isel({'Frame': start_frame, 'Cell Index': cell_ids[0]})),
                               axis=0)
                end_coord = np.stack((coords_ds['X'].isel({'Frame': end_frame, 'Cell Index': cell_ids[-1]}),
                               coords_ds['Y'].isel({'Frame': end_frame, 'Cell Index': cell_ids[-1]})),
                               axis=0)
                
                start_frames.append(start_frame)
                end_frames.append(end_frame)
                start_coords.append(start_coord)
                end_coords.append(end_coord)

        return (
            np.stack(start_frames, axis=0),
            np.stack(start_coords, axis=0),
            np.stack(end_frames, axis=0),
            np.stack(end_coords, axis=0),
            )

    def join_tracklets(self, max_time_gap: int = SETTINGS.FRAME_MEMORY, max_dist: int = SETTINGS.MAXIMUM_DISTANCE_THRESHOLD) -> None:
        with h5py.File(self.file, 'r+') as f:
            f[self.cells_group].attrs['maximum time gap in tracks'] = max_time_gap
        
        tracks_ids = np.unique(self.track_id_map[self.track_id_map >= 0])
        start_frames, start_coords, end_frames, end_coords = self.get_tracklet_endpoints()

        print('\nJoining tracklets...\n')

        dist_weights = np.linalg.norm(np.expand_dims(start_coords, 0) - np.expand_dims(end_coords, 1), axis=2)
        time_weights = np.expand_dims(start_frames, 0) - np.expand_dims(end_frames, 1)

        weights = np.where((time_weights>0) & (time_weights<max_time_gap) & (dist_weights<(max_dist*time_weights)), dist_weights, np.inf)
        potential_idxs_0, potential_idxs_1 = ~np.all(np.isinf(weights), axis=1), ~np.all(np.isinf(weights), axis=0)

        potential_dist_weights = dist_weights[potential_idxs_0][:, potential_idxs_1]
        potential_time_weights = time_weights[potential_idxs_0][:, potential_idxs_1]

        old, new = linear_sum_assignment(potential_dist_weights)
        valid_pairs = (potential_dist_weights[old, new] < max_dist) & (potential_time_weights[old, new] < max_time_gap) & (potential_time_weights[old, new] > 0)

        old, new = old[valid_pairs], new[valid_pairs]
        tracks_0 = tracks_ids[potential_idxs_0][old]
        tracks_1 = tracks_ids[potential_idxs_1][new]

        queue = np.column_stack((tracks_0, tracks_1))

        for i, (track_0, track_1) in enumerate(tqdm(queue)):
            # sys.stdout.write(f'\rTracklet {i + 1}/{len(queue)}')
            # sys.stdout.flush()
            # update track_id_map
            self.track_id_map[self.track_id_map == track_1] = track_0
            # update queue
            queue[:, 0][queue[:, 0]==track_1] = track_0

    def join_tracklets_batches(self, max_time_gap: int = SETTINGS.FRAME_MEMORY, max_dist: int = SETTINGS.MAXIMUM_DISTANCE_THRESHOLD, n_batches: int = 4, grid_size: int = 2) -> None:
        """
        Perform tracklet joining in smaller batches to save memory.
        Kmeans clustering to batch based on similar spatial coords.
        """
        with h5py.File(self.file, 'r+') as f:
            f[self.cells_group].attrs['maximum time gap in tracks'] = max_time_gap

        # === Step 1: Get endpoints for all tracklets ===
        tracks_ids = np.unique(self.track_id_map[self.track_id_map >= 0])
        start_frames, start_coords, end_frames, end_coords = self.get_tracklet_endpoints()

        # === Step 2: Cluster tracks based on start_coords ===
        print('\nClustering tracklets for batch joining ...')
        x_min, y_min = start_coords.min(axis=0)
        x_max, y_max = start_coords.max(axis=0)

        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)

        # Assign to grid cells
        x_idxs = np.digitize(start_coords[:, 0], x_bins) - 1
        y_idxs = np.digitize(start_coords[:, 1], y_bins) - 1

        # (x_idx, y_idx) is the grid cell
        grid_indices = list(zip(x_idxs, y_idxs))

        # Build cell-to-track mapping
        grid_to_track_ids = defaultdict(list)
        for i, cell in enumerate(grid_indices):
            grid_to_track_ids[cell].append(tracks_ids[i])
        # kmeans = KMeans(n_clusters=min(n_batches, len(tracks_ids)))
        # cluster_labels = kmeans.fit_predict(start_coords)

        # cluster_to_track_ids = defaultdict(list)
        # for i, cluster_id in enumerate(cluster_labels):
        #     cluster_to_track_ids[cluster_id].append(tracks_ids[i])


        def join_subset(track_ids_subset):
            track_ids_subset = np.asarray(track_ids_subset)
            id_to_idx = {tid: np.where(tracks_ids == tid)[0][0] for tid in track_ids_subset}
            indices = np.array([id_to_idx[tid] for tid in track_ids_subset])
            local_start_frames = start_frames[indices]
            local_start_coords = start_coords[indices]
            local_end_frames = end_frames[indices]
            local_end_coords = end_coords[indices]

            dist_weights = np.linalg.norm(
                np.expand_dims(local_start_coords, 0) - np.expand_dims(local_end_coords, 1), axis=2
            )
            time_weights = np.expand_dims(local_start_frames, 0) - np.expand_dims(local_end_frames, 1)

            weights = np.where(
                (time_weights > 0) & 
                (time_weights < max_time_gap) & 
                (dist_weights < (max_dist * time_weights)), 
                dist_weights, np.inf
            )

            potential_idxs_0 = ~np.all(np.isinf(weights), axis=1)
            potential_idxs_1 = ~np.all(np.isinf(weights), axis=0)

            potential_dist_weights = dist_weights[potential_idxs_0][:, potential_idxs_1]
            potential_time_weights = time_weights[potential_idxs_0][:, potential_idxs_1]

            if potential_dist_weights.size == 0:
                return

            old, new = linear_sum_assignment(potential_dist_weights)
            valid_pairs = (
                (potential_dist_weights[old, new] < max_dist) &
                (potential_time_weights[old, new] < max_time_gap) &
                (potential_time_weights[old, new] > 0)
            )

            old, new = old[valid_pairs], new[valid_pairs]
            tracks_0 = track_ids_subset[potential_idxs_0][old]
            tracks_1 = track_ids_subset[potential_idxs_1][new]

            queue = np.column_stack((tracks_0, tracks_1))

            for i, (track_0, track_1) in enumerate(queue):
                self.track_id_map[self.track_id_map == track_1] = track_0
                # update queue
                queue[:, 0][queue[:, 0]==track_1] = track_0

        
        print('\nJoining tracklets in batches...\n')

        # === Step 3: Process each cluster ===
        for i, (cluster_id, tids) in enumerate(tqdm(grid_to_track_ids.items())):
            # sys.stdout.write(f'\rBatch {i + 1}/{len(grid_to_track_ids)}')
            # sys.stdout.flush()
            join_subset(tids)

        # === Step 3: Final global join on reduced set ===
        print('\nFinal global join...\n')
        self.remove_short_tracks(min_track_length=3)
        self.remap_track_id_map()
        tracks_ids = np.unique(self.track_id_map[self.track_id_map >= 0])
        start_frames, start_coords, end_frames, end_coords = self.get_tracklet_endpoints()
        join_subset(tracks_ids)

    def remove_short_tracks(self, min_track_length: int = SETTINGS.MINIMUM_TRACK_LENGTH) -> None:
        """Remove any tracks with less than {min_track_length} instances"""
        track_ids, counts = np.unique(self.track_id_map[self.track_id_map >= 0], return_counts=True)
        short_track_ids = track_ids[counts <= min_track_length]

        mask = np.isin(self.track_id_map, short_track_ids)
        self.track_id_map[mask] = -1
        self.remap_track_id_map()

def main():
    my_tracker = Tracker()
    my_tracker.track()