import h5py
import numpy as np
from pathlib import Path
import pandas as pd
import sys
from scipy.optimize import linear_sum_assignment
import torch

from PhagoPred.utils import tools
from PhagoPred import SETTINGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_tracklets_np(file=SETTINGS.DATASET, mode='Epi', min_dist_threshold=SETTINGS.MINIMUM_DISTANCE_THRESHOLD):
    with h5py.File(file, 'r+') as f:
        print('\nTracking...')
        cell_ds = f.create_dataset(f'Cells/{mode}', shape=(0,0,3), maxshape=(None,None,None), dtype=np.float32)
        cell_ds.attrs['minimum distance'] = min_dist_threshold
        cell_ds.attrs['features'] = ['x', 'y', 'area']
        x_mesh_grid, y_mesh_grid = np.meshgrid(np.arange(SETTINGS.IMAGE_SIZE[0]), np.arange(SETTINGS.IMAGE_SIZE[1]))
        x_mesh_grid, y_mesh_grid = np.expand_dims(x_mesh_grid, 2), np.expand_dims(y_mesh_grid, 2)
        for i, name in enumerate(list(f['Segmentations'][mode].keys())[:10]):
            sys.stdout.write(f'\rFrame {i + 1}/{SETTINGS.NUM_FRAMES}')
            sys.stdout.flush()
            mask = f['Segmentations'][mode][name][:]
            idxs = tools.unique_nonzero(mask)
            expanded_mask = (np.expand_dims(mask, 2) == np.expand_dims(idxs, (0,1)))
            areas = np.sum(expanded_mask, axis=(0,1))
            x_centres = np.sum(x_mesh_grid*expanded_mask, axis=(0,1)) / areas
            y_centres = np.sum(y_mesh_grid*expanded_mask, axis=(0,1)) / areas
            current_instances = pd.DataFrame(np.vstack((idxs, x_centres, y_centres, areas)).T, columns=['idx', 'x', 'y', 'area'])
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
                    # old_idxs = np.append(old_idxs, np.arange(np.max(old_idxs), np.max(old_idxs)+len(spare_current_idxs))+1).astype(int)
                    old_idxs = np.append(old_idxs, np.arange(np.max(old_instances['idx']), np.max(old_instances['idx'])+len(spare_current_idxs))+1).astype(int)
                lut = np.zeros(np.max(current_idxs)+1)
                lut[current_idxs] = old_idxs
                # update mask with pixel mapping
                new_mask = lut[mask]
                f['Segmentations'][mode][name][...] = new_mask
                # update instance dataframe with pixel mapping
                current_instances.loc[:, 'idx'] = lut[np.array(current_instances.loc[:, 'idx']).astype(int)]
            # create dataset for each cell with centres info
            cell_idxs = current_instances.loc[:, 'idx']

            num_new_cells = max(np.max(cell_idxs)+1, cell_ds.shape[1]) - cell_ds.shape[1]
            if num_new_cells > 0:
                cell_ds.resize((cell_ds.shape[0], cell_ds.shape[1]+num_new_cells, cell_ds.shape[2]))
                cell_ds[:,-int(num_new_cells):] = np.full((cell_ds.shape[0], int(num_new_cells), cell_ds.shape[2]), np.nan)
            
            cell_ds.resize((cell_ds.shape[0]+1, cell_ds.shape[1], cell_ds.shape[2]))
            new_row = np.full((1, cell_ds.shape[1], cell_ds.shape[2]), np.nan)
            for _, instance in current_instances.iterrows():
                new_row[:, int(instance['idx'])] = [instance['x'], instance['y'], instance['area']]

            cell_ds[-1] = new_row
            old_instances = current_instances

            old_instances = current_instances

def get_tracklets_torch(file=SETTINGS.DATASET, mode='Epi', min_dist_threshold=SETTINGS.MINIMUM_DISTANCE_THRESHOLD, cell_batch_size=50):
    with h5py.File(file, 'r+') as f:
        print('\nTracking...')
        cell_ds = f.create_dataset(f'Cells/{mode}', shape=(0,0,3), maxshape=(None,None,None), dtype=np.float32)
        # cell_ds = f[f'Cells/{mode}']
        cell_ds.attrs['minimum distance'] = min_dist_threshold
        cell_ds.attrs['features'] = ['x', 'y', 'area']
        x_mesh_grid, y_mesh_grid = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).to(device), torch.arange(SETTINGS.IMAGE_SIZE[1]).to(device), indexing='ij')
        x_mesh_grid, y_mesh_grid = x_mesh_grid.unsqueeze(2), y_mesh_grid.unsqueeze(2)
        for i, name in enumerate(list(f['Segmentations'][mode].keys())):
            sys.stdout.write(f'\rFrame {i + 1}/{SETTINGS.NUM_FRAMES}')
            sys.stdout.flush()
            mask = torch.tensor(f['Segmentations'][mode][name][:]).to(device)
            idxs = torch.unique(mask[mask != 0])
            areas_list = []
            x_centres_list = []
            y_centres_list = []

            for start_idx in range(0, len(idxs), cell_batch_size):
                last_idx = min(start_idx+cell_batch_size, len(idxs))
                batch_idxs = idxs[start_idx:last_idx]

                # Expand the mask -> each cell assigned own slice in dim 2
                expanded_mask = mask.unsqueeze(2) == batch_idxs.unsqueeze(0).unsqueeze(1)

                # Compute areas
                areas_batch = torch.sum(expanded_mask, dim=(0, 1))
                areas_list.append(areas_batch.cpu())

                # Compute centres
                x_centres_batch = torch.sum(x_mesh_grid * expanded_mask, dim=(0, 1)) / areas_batch
                y_centres_batch = torch.sum(y_mesh_grid * expanded_mask, dim=(0, 1)) / areas_batch

                x_centres_list.append(x_centres_batch.cpu())
                y_centres_list.append(y_centres_batch.cpu())

            # Concatenate results from all batches
                areas = torch.cat(areas_list)
                x_centres = torch.cat(x_centres_list).numpy()
                y_centres = torch.cat(y_centres_list).numpy()

            # Create a DataFrame with the results
            current_instances = pd.DataFrame(
                np.vstack((idxs.cpu().numpy(), y_centres, x_centres, areas)).T,
                columns=['idx', 'x', 'y', 'area']
            ).astype(dtype={'idx': 'int16', 'x': 'float32', 'y': 'float32', 'area': 'float32'})
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
                    # old_idxs = np.append(old_idxs, np.arange(np.max(old_idxs), np.max(old_idxs)+len(spare_current_idxs))+1).astype(int)
                    old_idxs = np.append(old_idxs, np.arange(np.max(old_instances['idx']), np.max(old_instances['idx'])+len(spare_current_idxs))+1).astype(int)
                lut = np.zeros(np.max(current_idxs)+1)
                lut[current_idxs] = old_idxs
                # update mask with pixel mapping
                new_mask = lut[mask.cpu().numpy()]
                f['Segmentations'][mode][name][...] = new_mask
                # update instance dataframe with pixel mapping
                current_instances.loc[:, 'idx'] = lut[np.array(current_instances.loc[:, 'idx']).astype(int)]
            # create dataset for each cell with centres info
            cell_idxs = current_instances.loc[:, 'idx']

            num_new_cells = max(np.max(cell_idxs)+1, cell_ds.shape[1]) - cell_ds.shape[1]
            if num_new_cells > 0:
                cell_ds.resize((cell_ds.shape[0], cell_ds.shape[1]+num_new_cells, cell_ds.shape[2]))
                cell_ds[:,-int(num_new_cells):] = np.full((cell_ds.shape[0], int(num_new_cells), cell_ds.shape[2]), np.nan)
            
            cell_ds.resize((cell_ds.shape[0]+1, cell_ds.shape[1], cell_ds.shape[2]))
            new_row = np.full((1, cell_ds.shape[1], cell_ds.shape[2]), np.nan)
            for _, instance in current_instances.iterrows():
                new_row[:, int(instance['idx'])] = [instance['x'], instance['y'], instance['area']]

            cell_ds[-1] = new_row
            old_instances = current_instances
        

def get_start_end_frames(f, mode, get_coords=True):
    x_coords = tools.get_features_ds(f['Cells'][mode], 'x')
    y_coords = tools.get_features_ds(f['Cells'][mode], 'y')
    coords = np.stack((x_coords, y_coords), axis=2)

    not_nan = ~np.isnan(coords[:,:,0])

    start_frames = np.argmax(not_nan, axis=0)
    end_frames = len(not_nan) - 1 - np.argmax(not_nan[::-1], axis=0)
    if get_coords:
        start_coords = coords[start_frames, np.arange(len(start_frames))]
        end_coords = coords[end_frames, np.arange(len(end_frames))]
        print(start_coords.shape)
        return start_frames, start_coords, end_frames, end_coords
    else:
        return start_frames, end_frames
    
def join_tracklets(file=SETTINGS.DATASET, mode='Epi', time_threshold=SETTINGS.FRAME_MEMORY, distance_threshold=SETTINGS.MINIMUM_DISTANCE_THRESHOLD, min_track_length=SETTINGS.MINIMUM_TRACK_LENGTH):
    with h5py.File(file, 'r+') as f:
        print('\nJoining Tracklets...')
        f['Cells'][mode][:, 0] = np.zeros(shape=f['Cells'][mode][:, 0][...].shape)
        f['Cells'][mode].attrs['minimum time gap'] = time_threshold
        f['Cells'][mode].attrs['minimum track length'] = min_track_length
        all_cell_idxs = np.arange(len(f['Cells'][mode][0][:]))

        start_frames, start_coords, end_frames, end_coords = get_start_end_frames(f, mode)

        dist_weights = np.linalg.norm(np.expand_dims(start_coords, 0) - np.expand_dims(end_coords, 1), axis=2)
        time_weights = np.expand_dims(start_frames, 0) - np.expand_dims(end_frames, 1)

        weights = np.where((time_weights>0) & (time_weights<time_threshold) & (dist_weights<(distance_threshold*time_weights)), dist_weights, np.inf)
        potential_idxs_0, potential_idxs_1 = ~np.all(np.isinf(weights), axis=1), ~np.all(np.isinf(weights), axis=0)

        potential_dist_weights = dist_weights[potential_idxs_0][:, potential_idxs_1]
        potential_time_weights = time_weights[potential_idxs_0][:, potential_idxs_1]

        old, new = linear_sum_assignment(potential_dist_weights)
        valid_pairs = (potential_dist_weights[old, new] < distance_threshold) & (potential_time_weights[old, new] < time_threshold) & (potential_time_weights[old, new] > 0)

        old, new = old[valid_pairs], new[valid_pairs]
        cells_0 = all_cell_idxs[potential_idxs_0][old]
        cells_1 = all_cell_idxs[potential_idxs_1][new]

        # join new_cells to old_cells, and adjust segmentations. Will need to repack hdf5 after, to clear space from unused datasets. Must also update cells0, cells1 list wiith new idxs.
        queue = np.column_stack((cells_0, cells_1))
        for i, (cell_0, cell_1) in enumerate(queue):
            sys.stdout.write(f'\rProgress {int(((i + 1)/len(queue))*100):03}%')
            sys.stdout.flush()
            # update Cells group
            cell_1_ds = f['Cells'][mode][start_frames[cell_1]:, cell_1]
            f['Cells'][mode][-len(cell_1_ds):, cell_0] = cell_1_ds
            # update queue
            queue[:, 0][queue[:, 0]==cell_1] = cell_0
            # update masks
            for frame in range(start_frames[cell_1], end_frames[cell_1]+1):
                mask = f['Segmentations'][mode][f'{int(frame):04}'][:]
                mask[mask==cell_1] = cell_0
                f['Segmentations'][mode][f'{int(frame):04}'][...] = mask
            f['Cells'][mode][:, cell_1] = np.full(f['Cells'][mode][:, cell_1].shape, np.nan)

        print('\nRemoving short tracks...')
        start_frames, end_frames = get_start_end_frames(f, mode, get_coords=False)
        track_lengths = end_frames - start_frames
        for cell_idx in range(f['Cells'][mode].shape[1]):
            if track_lengths[cell_idx] < min_track_length:
                f['Cells'][mode][:, cell_idx] = np.full(f['Cells'][mode][:, cell_idx].shape, np.nan)
                for frame in range(start_frames[cell_idx], end_frames[cell_idx]+1):
                    mask = f['Segmentations'][mode][f'{int(frame):04}'][:]
                    mask[mask==cell_idx] = 0
                    f['Segmentations'][mode][f'{int(frame):04}'][...] = mask
        

def remove_empty_cells(file=SETTINGS.DATASET, mode='Epi'):
    print('\nRemoving empty tracks and reindexing...')
    with h5py.File(file, 'r+') as f:
        old_cell_idxs = np.nonzero(np.any(~np.isnan(f['Cells'][mode][:]), axis=(0,2)))[0]
        new_cell_idxs = np.arange(len(old_cell_idxs))
        lut = np.zeros(f['Cells'][mode].shape[1])
        lut[old_cell_idxs] = new_cell_idxs
        # update masks
        print('\nUpdating masks...')
        for frame in range(f['Cells'][mode].shape[0]):
            mask = f['Segmentations'][mode][f'{int(frame):04}'][:]
            mask = lut[mask]
            f['Segmentations'][mode][f'{int(frame):04}'][...] = mask
        # squash dataset
        print('\nSquashing dataset...')
        ds = f['Cells'][mode][:]
        squashed_ds = ds[:, old_cell_idxs]
        f['Cells'][mode].resize(squashed_ds.shape)
        f['Cells'][mode][()] = squashed_ds
        


def main():
    for mode in ['Phase']:
        print('\nMode: ', mode)
        if device.type=='cpu':
            get_tracklets_np(mode=mode)
        else:
            get_tracklets_torch(mode=mode)
        join_tracklets(mode=mode)
        remove_empty_cells(mode=mode)
    print('\nRepacking hdf5 file...')
    tools.repack_hdf5()


if __name__=='__main__':
    main()
