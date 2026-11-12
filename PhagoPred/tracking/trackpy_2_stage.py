import h5py
import numpy as np
import trackpy as tp
from tqdm import tqdm
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction import extract_features, features


class TrackerWithTrackpy2Stage:

    def __init__(self, file_path, channel='Phase'):
        self.file = file_path
        self.channel = channel
        self.masks_ds = f'Segmentations/{self.channel}'
        self.cells_group = f'Cells/{self.channel}'
        self.max_track_id = 0
        self.track_id_map = None  # will be a 2D array: frames x max_cells
        self.min_cell_size = 500

    def extract_features_for_tracking(self):
        # Load coordinates (X, Y) for all cells in all frames
        with h5py.File(self.file, 'r') as f:
            coords = []
            num_frames = f[self.masks_ds].shape[0]
            for frame in tqdm(range(num_frames),
                              desc='Gathering cell coords: '):
                X = f[f'{self.cells_group}/X'][frame]
                Y = f[f'{self.cells_group}/Y'][frame]
                areas = f[f'{self.cells_group}/Area'][frame]

                valid_idxs = np.nonzero(~np.isnan(X) & ~np.isnan(Y)
                                        & (areas > self.min_cell_size))[0]

                for idx in valid_idxs:
                    coords.append({
                        'frame': frame,
                        'x': X[idx],
                        'y': Y[idx],
                        'area': areas[idx],
                        'cell_index': idx
                    })

        return coords, num_frames

    def track(self,
              search_range=10,
              memory=3,
              stage2_search_range=None,
              stage2_memory=None,
              area_scale=45,
              max_overlap=5):
        """
        Two-stage tracking:
          Stage 1: Link with small search_range to build reliable short tracklets.
          Stage 2: Represent each tracklet as a single point (last known position)
                   and link tracklets with a larger search_range.

        Args:
            search_range: Small search range for stage 1 (fast, reliable links)
            memory: Frame memory for stage 1
            stage2_search_range: Larger search range for stage 2 (default: 3x search_range)
            stage2_memory: Frame memory for stage 2 (default: 3x memory)
        """
        if stage2_search_range is None:
            stage2_search_range = search_range * 3
        if stage2_memory is None:
            stage2_memory = memory * 3

        coords, num_frames = self.extract_features_for_tracking()
        coords = pd.DataFrame(coords)
        coords['log_area'] = area_scale * np.log(coords['area'])

        # --- Stage 1: fine-grained linking with small search_range ---
        print(
            f'\nStage 1: Linking with search_range={search_range}, memory={memory}'
        )
        df = tp.link_df(coords,
                        search_range=search_range,
                        memory=memory,
                        pos_columns=['x', 'y', 'log_area'])
        n_tracklets = df['particle'].nunique()
        print(f'Stage 1 produced {n_tracklets} tracklets')

        # --- Stage 2: link tracklets using their endpoint positions ---
        print(
            f'\nStage 2: Linking tracklets with search_range={stage2_search_range}, memory={stage2_memory}'
        )
        df = self._link_tracklets(
            df, stage2_search_range, stage2_memory, 0, max_overlap
        )  # Use area sclae = 0 for tracklet linking. Cell often change size significantly when dying.
        n_tracks = df['particle'].nunique()
        print(f'Stage 2 merged into {n_tracks} tracks')

        # --- Build track_id_map ---
        self.max_track_id = df['particle'].max()
        # Size track_id_map to cover ALL cell indices in the masks, not just tracked ones
        with h5py.File(self.file, 'r') as f:
            max_cells = f[self.cells_group + '/X'].shape[1]
        self.track_id_map = -np.ones((num_frames, max_cells), dtype=int)

        for row in df.itertuples():
            self.track_id_map[row.frame, row.cell_index] = row.particle

        with h5py.File(self.file, 'r+') as f:
            f['Segmentations']['Phase'].attrs[
                'Tracking search range'] = search_range
            f['Segmentations']['Phase'].attrs[
                'Tracking search range stage 2'] = stage2_search_range
            f['Segmentations']['Phase'].attrs['Tracking memory'] = memory

    def _link_tracklets(self, df, max_dist, max_time_gap, area_scale,
                        max_overlap):
        """
        Match END positions of tracklets to START positions of other tracklets
        using KDTree for spatial lookup + Hungarian algorithm for optimal matching.
        Memory-efficient: avoids building full N x N distance matrix.
        """
        # Build tracklet endpoints
        info = df.groupby('particle').agg(
            first_frame=('frame', 'min'),
            last_frame=('frame', 'max'),
        )
        first_rows = df.loc[df.groupby('particle')
                            ['frame'].idxmin()].set_index('particle')
        last_rows = df.loc[df.groupby('particle')['frame'].idxmax()].set_index(
            'particle')

        all_tracklet_ids = info.index.values
        start_frames = info['first_frame'].values
        end_frames = info['last_frame'].values
        start_coords = np.column_stack([
            first_rows.loc[info.index, 'x'].values, first_rows.loc[info.index,
                                                                   'y'].values
        ])
        end_coords = np.column_stack([
            last_rows.loc[info.index, 'x'].values, last_rows.loc[info.index,
                                                                 'y'].values
        ])
        start_areas = first_rows.loc[info.index, 'area'].values
        end_areas = last_rows.loc[info.index, 'area'].values

        # Use KDTree on start_coords to find spatial neighbours for each end_coord
        # max possible distance is max_dist * max_time_gap (distance scales with time gap)
        spatial_radius = max_dist * np.sqrt(max_time_gap)
        start_tree = cKDTree(start_coords)

        # Collect valid (end_idx, start_idx, distance, |log area ratio|) pairs
        tracks = {
            pid: (sub['frame'].to_numpy(), sub[['x', 'y']].to_numpy(),
                  sub['area'].to_numpy())
            for pid, sub in df.sort_values('frame').groupby('particle')
        }

        candidate_pairs = []
        for i, end_coord in enumerate(end_coords):
            e = end_frames[i]
            frames_i = tracks[all_tracklet_ids[i]][0]
            for j in start_tree.query_ball_point(end_coord, r=spatial_radius):
                if end_frames[j] <= e or start_frames[j] - e > max_time_gap:
                    continue
                frames_j, xy_j, area_j = tracks[all_tracklet_ids[j]]
                if start_frames[j] <= e:
                    # j began before i ended: only the same cell if they took
                    # turns (segmentation flicker), never both in one frame
                    if (e - start_frames[j] >= max_overlap or np.isin(
                            frames_j[frames_j <= e], frames_i).any()):
                        continue
                k = np.searchsorted(
                    frames_j, e, side='right')  # j's first detection after e
                time_gap = frames_j[k] - e
                if time_gap > max_time_gap:
                    continue
                dist = np.linalg.norm(end_coord - xy_j[k])
                log_area_ratio = abs(np.log(end_areas[i] / area_j[k]))
                step = (np.hypot(dist, area_scale * log_area_ratio) /
                        np.sqrt(time_gap))
                if step < max_dist:
                    candidate_pairs.append((i, j, step, time_gap))

        if not candidate_pairs:
            return df

        candidate_pairs = np.array(candidate_pairs)
        end_idxs = candidate_pairs[:, 0].astype(int)
        start_idxs = candidate_pairs[:, 1].astype(int)
        steps = candidate_pairs[:, 2]
        # log_area_ratios = candidate_pairs[:, 3]
        # gaps = start_frames[start_idxs] - end_frames[end_idxs]

        unique_ends = np.unique(end_idxs)
        unique_starts = np.unique(start_idxs)
        end_remap = {v: k for k, v in enumerate(unique_ends)}
        start_remap = {v: k for k, v in enumerate(unique_starts)}

        gaps = candidate_pairs[:, 3]
        cost_matrix = np.zeros((len(unique_ends), len(unique_starts)))
        for ei, si, step, gap in zip(end_idxs, start_idxs, steps, gaps):
            cost_matrix[end_remap[ei],
                        start_remap[si]] = (step - max_dist) / gap

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        valid = cost_matrix[row_ind, col_ind] < 0
        row_ind, col_ind = row_ind[valid], col_ind[valid]

        merge_from = all_tracklet_ids[
            unique_ends[row_ind]]  # end tracklets (earlier)
        merge_into = all_tracklet_ids[
            unique_starts[col_ind]]  # start tracklets (later)

        merge_map = dict(zip(merge_into, merge_from))

        # Resolve chains
        def resolve(tid):
            while tid in merge_map:
                tid = merge_map[tid]
            return tid

        df = df.copy()
        df['particle'] = df['particle'].map(resolve)

        # Re-number consecutively
        unique_ids = df['particle'].unique()
        remap = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(unique_ids))
        }
        df['particle'] = df['particle'].map(remap)

        return df

    def apply_track_id_map(self):
        with h5py.File(self.file, 'r+') as f:

            print('\nUpdating datasets with track ID map...\n')
            required_width = max(self.max_track_id + 1,
                                 self.track_id_map.shape[1])
            for feature_data in f[self.cells_group].values():
                if required_width > feature_data.shape[1]:
                    feature_data.resize(required_width, axis=1)

            for frame, track_id_map in enumerate(tqdm(self.track_id_map)):
                mask = f[self.masks_ds][frame][:]
                # Replace mask indices with track IDs, keep -1 for background
                f[self.masks_ds][frame] = np.where(mask == -1, -1,
                                                   track_id_map[mask])

                valid_cell_idxs = np.nonzero(track_id_map >= 0)[0]
                valid_track_id_map = track_id_map[valid_cell_idxs]

                for feature_name, feature_data in f[self.cells_group].items():
                    feature_vals = feature_data[frame][valid_cell_idxs]
                    reindexed_feature_data = np.full(self.max_track_id + 1,
                                                     np.nan)
                    reindexed_feature_data[valid_track_id_map] = feature_vals
                    f[self.cells_group][feature_name][frame, :len(
                        reindexed_feature_data)] = reindexed_feature_data

            for feature_data in f[self.cells_group].values():
                feature_data.resize(self.max_track_id + 1, axis=1)

    def remove_short_tracks(
            self,
            min_track_length: int = SETTINGS.MINIMUM_TRACK_LENGTH) -> None:
        """Remove any tracks with less than {min_track_length} instances"""
        track_ids, counts = np.unique(
            self.track_id_map[self.track_id_map >= 0], return_counts=True)
        short_track_ids = track_ids[counts <= min_track_length]

        mask = np.isin(self.track_id_map, short_track_ids)
        self.track_id_map[mask] = -1
        self.remap_track_id_map()

    def remap_track_id_map(self):
        """Reindex the track_id_map so all track_ids are consecutive"""
        valid_mask = self.track_id_map >= 0
        track_ids = np.unique(self.track_id_map[valid_mask])
        new_track_ids = np.arange(len(track_ids))

        id_map = -np.ones(track_ids.max() + 1, dtype=int)
        id_map[track_ids] = new_track_ids

        self.track_id_map = np.where(valid_mask, id_map[self.track_id_map], -1)

        self.max_track_id = np.max(self.track_id_map)

    def get_cell_info(self):
        """
        Add each cells centroid cooridnates and areas to cells dataset
        """
        extract_features.extract_features(self.file, features.Coords())


def run_tracking(dataset=SETTINGS.DATASET):
    tracker = TrackerWithTrackpy2Stage(dataset, 'Phase')

    tracker.get_cell_info()
    tracker.track(
        search_range=SETTINGS.STAGE_1_RANGE,
        stage2_search_range=SETTINGS.STAGE_2_RANGE,
        memory=SETTINGS.STAGE_1_MEMORY,
        stage2_memory=SETTINGS.STAGE_2_MEMORY,
        area_scale=SETTINGS.TRACK_AREA_SCALE,
        max_overlap=SETTINGS.TRACK_MAX_OVERLAP,
    )
    tracker.remove_short_tracks(SETTINGS.MINIMUM_TRACK_LENGTH)
    tracker.apply_track_id_map()


def main():
    run_tracking('/home/ubuntu/PhagoPred/PhagoPred/Datasets/A1.h5')


if __name__ == "__main__":
    main()
