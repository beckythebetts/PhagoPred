import numpy as np
import torch
import h5py
import sys
from pathlib import Path
import imageio

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs
from PhagoPred.tracking import nearest_neighbour_tracking


def detect_events(dataset=SETTINGS.DATASET, batch_size=50):
    x_grid, y_grid = np.meshgrid(np.arange(SETTINGS.IMAGE_SIZE[0]), np.arange(SETTINGS.IMAGE_SIZE[1]))
    with h5py.File(dataset, 'r+') as f:
        phago_events = f.create_group('Phagocytosis')
        num_batches = (SETTINGS.NUM_FRAMES + batch_size - 1) // batch_size
        dtype = np.dtype([('frame', np.int16),
                          ('pathogen index', np.int16), 
                          ('pixel count', np.uint8),
                          ('x centre', np.float32),
                          ('y centre', np.float32)])
        for i in range(num_batches):
            first_frame = i*batch_size
            last_frame = min((i+1)*batch_size, SETTINGS.NUM_FRAMES)
            print('frames:', first_frame, last_frame)
            phase_masks = tools.get_masks(f, 'Phase', first_frame, last_frame)
            epi_masks = tools.get_masks(f, 'Epi', first_frame, last_frame)
            overlaps = np.logical_and(phase_masks, epi_masks)
            overlap_idxs = phase_masks*overlaps
            for cell_idx in tools.unique_nonzero(overlap_idxs, return_counts=False):
                phago_list = []
                if f'{cell_idx:04}' not in phago_events.keys():
                    events_dataset = phago_events.create_dataset(f'{cell_idx:04}/all', shape=(0,), maxshape=(None,), dtype=dtype)
                else:
                    events_dataset = phago_events[f'{cell_idx:04}']['all']
                cell_mask = overlap_idxs == cell_idx
                pathogen_idxs = cell_mask*epi_masks
                pathogen_idxs = [tools.unique_nonzero(slice, return_counts=True) for slice in pathogen_idxs]
                for frame_idx, (idxs, counts) in enumerate(pathogen_idxs):
                    if len(idxs) > 0:
                        for idx, count in zip(idxs, counts):
                            x, y = mask_funcs.get_centre(epi_masks[frame_idx]==idx, x_mesh_grid=x_grid, y_mesh_grid=y_grid)
                            phago_list.append((frame_idx+first_frame, idx, count, x, y))
                events_dataset.resize(len(events_dataset)+len(phago_list), axis=0)
                events_dataset[-len(phago_list):] = np.array(phago_list, dtype=dtype)
            

def track_pathogens(dataset=SETTINGS.DATASET):
    # Need to also track pathogens before and after phagocytosis
    dtype = dtype = np.dtype([('frame', 'f4'), ('pathogen index', 'f4'), ('pathogen x', 'f4'), ('pathogen y', 'f4')])
    with h5py.File(dataset, 'r+') as f:
        for cell in f['Phagocytosis'].keys():
            dataset = f['Phagocytosis'][cell]['all']
            tracker = nearest_neighbour_tracking.NearestNeighbourTracking(dataset['frame'][:],
                                                                          dataset['pathogen index'][:],
                                                                          np.column_stack((dataset['x centre'][:], dataset['y centre'][:])))
            tracker.track()
            for pathogen_track in tracker.tracked:  
                frames, idxs = np.fromiter(pathogen_track.track_dict.keys(), dtype=np.int16), np.fromiter(pathogen_track.track_dict.values(), dtype=np.int16)
                print(frames, idxs)
                track_dataset = f['Phagocytosis'][cell].create_dataset(f'{frames[0]}_{frames[-1]}', shape=(SETTINGS.NUM_FRAMES,), maxshape=(None,), dtype=dtype)
                # fill in any gaps with np.nan
                frames_full = np.arange(0, SETTINGS.NUM_FRAMES)
                idxs_full = np.full(frames_full.shape, np.nan)
                idxs_full[np.isin(idxs_full, frames)] = idxs
                track_dataset[:] = np.array([(frames_full, idxs_full)], dtype=dtype)
            
                


def main():
    #detect_events()
    track_pathogens()

if __name__ == '__main__':
    main()