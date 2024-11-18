import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from scipy.optimize import linear_sum_assignment
import networkx as nx
import matplotlib.pyplot as plt

from PhagoPred.utils import tools, mask_funcs
from PhagoPred import SETTINGS


def get_tracklets(file=SETTINGS.DATASET, mode='Epi', min_dist_threshold=10):
    with h5py.File(file, 'r+') as f:
        cells_group = f.create_group(f'Cells/{mode}')
        dtype = np.dtype([('frame', 'f4'), 
                          ('x', 'f4'),
                          ('y', 'f4')])
        x_mesh_grid, y_mesh_grid = np.meshgrid(np.arange(SETTINGS.IMAGE_SIZE[0]), np.arange(SETTINGS.IMAGE_SIZE[1]))
        x_mesh_grid, y_mesh_grid = np.expand_dims(x_mesh_grid, 2), np.expand_dims(y_mesh_grid, 2)
        for i, name in enumerate(list(f['Segmentations'][mode].keys())):
            sys.stdout.write(f'\rFrame {i + 1}/{SETTINGS.NUM_FRAMES}')
            sys.stdout.flush()
            mask = f['Segmentations'][mode][name][:]
            idxs = tools.unique_nonzero(mask)
            expanded_mask = (np.expand_dims(mask, 2) == np.expand_dims(idxs, (0,1)))
            areas = np.sum(expanded_mask, axis=(0,1))
            x_centres = np.sum(x_mesh_grid*expanded_mask, axis=(0,1)) / areas
            y_centres = np.sum(y_mesh_grid*expanded_mask, axis=(0,1)) / areas
            current_instances = pd.DataFrame(np.vstack((idxs, x_centres, y_centres)).T, columns=['idx', 'x', 'y'])
            if i != 0:
                distances = np.linalg.norm(
                    np.expand_dims(old_instances.loc[:, 'x':'y'].values, 1) - np.expand_dims(current_instances.loc[:, 'x':'y'].values, 0), 
                    axis=2
                )
                old, new = linear_sum_assignment(distances)
                valid_pairs = distances[old, new] < min_dist_threshold
                old, new = old[valid_pairs], new[valid_pairs]
                old_idxs = np.array(old_instances.loc[:, 'idx']).astype(int)[old]
                current_idxs = np.array(current_instances.loc[:, 'idx']).astype(int)[new]
                if len(current_instances) > len(current_idxs):
                    spare_current_idxs = np.array(current_instances[~current_instances['idx'].isin(current_idxs)].loc[:, 'idx'])
                    current_idxs = np.append(current_idxs, spare_current_idxs).astype(int)
                    old_idxs = np.append(old_idxs, np.arange(np.max(old_idxs), np.max(old_idxs)+len(spare_current_idxs))+1).astype(int)
                lut = np.zeros(np.max(current_idxs)+1)
                lut[current_idxs] = old_idxs
                # update mask with pixel mapping
                new_mask = lut[mask]
                f['Segmentations'][mode][name][...] = new_mask
                # update instance dataframe with pixel mapping
                current_instances.loc[:, 'idx'] = lut[np.array(current_instances.loc[:, 'idx']).astype(int)]
            # create dataset for each cell with centres info

            for _, instance in current_instances.iterrows():
                dataset_name = f'{int(instance["idx"]):04}'
                if dataset_name not in cells_group:
                    cells_group.create_dataset(dataset_name, shape=(0,), dtype=dtype, maxshape=(None,))
                cell_ds = cells_group[dataset_name]
                cell_ds.resize((cell_ds.shape[0]+1, ))
                cell_ds[-1] = (i, instance['x'], instance['y'])
            old_instances = current_instances

def join_tracklets(file=SETTINGS.DATASET, mode='Epi', time_threshold=4, distance_threshold=10, min_track_length=30):
    with h5py.File(file, 'r+') as f:
        cell_idxs = np.array(list(f['Cells'][mode].keys()))
        start_coords = pd.DataFrame([list(f['Cells'][mode][cell_idx][0]) for cell_idx in cell_idxs], index=cell_idxs, columns=['frame', 'x', 'y'])
        end_coords = pd.DataFrame([list(f['Cells'][mode][cell_idx][-1]) for cell_idx in cell_idxs], index=cell_idxs, columns=['frame', 'x', 'y'])
        dist_weights = np.linalg.norm(np.expand_dims(start_coords.loc[:, 'x':'y'].values, 0) - np.expand_dims(end_coords.loc[:, 'x':'y'].values, 1), axis=2)
        time_weights = np.expand_dims(start_coords.loc[:, 'frame'].values, 0) - np.expand_dims(end_coords.loc[:, 'frame'].values, 1)
        weights = np.where((time_weights>0) & (time_weights<time_threshold) & (dist_weights<distance_threshold), dist_weights, np.inf)
        potential_idxs_0, potential_idxs_1 = ~np.all(np.isinf(weights), axis=1), ~np.all(np.isinf(weights), axis=0)
        potential_cells_0, potential_cells_1 = cell_idxs[potential_idxs_0], cell_idxs[potential_idxs_1]
        potential_dist_weights = dist_weights[potential_idxs_0][:, potential_idxs_1]
        potential_time_weights = time_weights[potential_idxs_0][:, potential_idxs_1]
        old, new = linear_sum_assignment(potential_dist_weights)
        valid_pairs = (potential_dist_weights[old, new] < distance_threshold) & (potential_time_weights[old, new] < time_threshold) & (potential_time_weights[old, new] > 0)
        old, new = old[valid_pairs], new[valid_pairs]
        cells_0 = potential_cells_0[old]
        cells_1 = potential_cells_1[new]
        # join new_cells to old_cells, and adjust segmentations. Will need to repack hdf5 after, to clear space from unused datasets. Must also update cells0, cells1 list wiith new idxs.
        queue = np.column_stack((cells_0, cells_1))
        for cell_0, cell_1 in queue:
            # update Cells group
            cell_0_ds, cell_1_ds = f['Cells'][mode][cell_0], f['Cells'][mode][cell_1]
            cell_0_ds.resize((len(cell_0_ds) + len(cell_1_ds), ))
            cell_0_ds[-len(cell_1_ds):] = cell_1_ds[:]
            # update queue
            queue[:, 0][queue[:, 0]==cell_1] = cell_0
            # update masks
            for frame in cell_1_ds['frame'][:]:
                mask = f['Segmentations'][mode][f'{int(frame):04}'][:]
                mask[mask==int(cell_1)] = cell_0
                f['Segmentations'][mode][f'{int(frame):04}'][...] = mask
            del f['Cells'][mode][cell_1]

        for cell in f['Cells'][mode].keys():
            if len(f['Cells'][mode][cell][:]) < min_track_length:
                for frame in f['Cells'][mode][cell]['frame'][:]:
                    mask = f['Segmentations'][mode][f'{int(frame):04}'][:]
                    mask[mask==int(cell)] = 0
                    f['Segmentations'][mode][f'{int(frame):04}'][...] = mask
                del f['Cells'][mode][cell]

def main():
    get_tracklets(mode='Phase')
    join_tracklets(mode='Phase')
    tools.repack_hdf5()

if __name__=='__main__':
    main()
