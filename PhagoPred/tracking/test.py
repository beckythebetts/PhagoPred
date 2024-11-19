import h5py
import torch
import numpy as np
import sys
import pandas as pd
import time

from PhagoPred import SETTINGS
from PhagoPred.utils import tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_torch(file = SETTINGS.DATASET, mode='Epi'):
    time0 = time.time()
    with h5py.File(file, 'r') as f:
        x_mesh_grid, y_mesh_grid = torch.meshgrid(torch.arange(SETTINGS.IMAGE_SIZE[0]).to(device), torch.arange(SETTINGS.IMAGE_SIZE[1]).to(device))
        x_mesh_grid, y_mesh_grid = x_mesh_grid.unsqueeze(2), y_mesh_grid.unsqueeze(2)
        for i, name in enumerate(list(f['Segmentations'][mode].keys())[:10]):
            sys.stdout.write(f'\rFrame {i + 1}/{SETTINGS.NUM_FRAMES}')
            sys.stdout.flush()
            mask = torch.tensor(f['Segmentations'][mode][name][:]).to(device)
            idxs = torch.unique(mask[mask != 0])
            expanded_mask = (mask.unsqueeze(2) == idxs.unsqueeze(0).unsqueeze(1))
            areas = torch.sum(expanded_mask, dim=(0,1))
            x_centres = (torch.sum(x_mesh_grid*expanded_mask, dim=(0,1)) / areas).cpu().numpy()
            y_centres = (torch.sum(y_mesh_grid*expanded_mask, dim=(0,1)) / areas).cpu().numpy()
            current_instances = pd.DataFrame(np.vstack((idxs.cpu().numpy(), y_centres, x_centres)).T, columns=['idx', 'x', 'y']).astype(dtype={'idx':'int16', 'x':'float32', 'y':'float32'})
    time1 = time.time()
    return current_instances, time1-time0

def test_numpy(file = SETTINGS.DATASET, mode='Epi'):
    time0 = time.time()
    with h5py.File(file, 'r') as f:
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
            current_instances = pd.DataFrame(np.vstack((idxs, x_centres, y_centres)).T, columns=['idx', 'x', 'y']).astype(dtype={'idx':'int16', 'x':'float32', 'y':'float32'})
    time1 = time.time()
    return current_instances, time1-time0

if __name__ == '__main__':
    torch_inst, torch_time = test_torch()
    np_inst, np_time = test_numpy()
    print('\n', torch_time, np_time)
    if torch_inst.equals(np_inst):
        print('same')