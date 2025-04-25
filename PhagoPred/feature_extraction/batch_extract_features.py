import h5py
import numpy as np
import torch
import sys
import skimage
import time
from pyefd import elliptic_fourier_descriptors
import matplotlib.pyplot as plt

from PhagoPred import SETTINGS
from PhagoPred.utils import tools, mask_funcs
from PhagoPred.feature_extraction.morphology import fitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def get_phagocytosis(file=SETTINGS.DATASET):
#     with h5py.File(file, 'r+') as f:
#         phago_group = f.create_group('Cells/Phagocytosis')
#         epi = tools.get_masks(f, 'Epi')
#         phase = tools.get_masks(f, 'Phase')
#         overlap = (epi>0) & (phase>0)
#         epi_idxs = epi[overlap]
#         phase_idxs = phase[overlap]
#         frames = np.nonzero(overlap)[0]
#         events, counts = np.unique(np.stack((phase_idxs, epi_idxs, frames), axis=1), axis=0, return_counts=True)
#         idx_pairs = events[:, :2]
#         frames = events[:, 2]
#         for idx_pair in np.unique(events[:, :2], axis=0):
#             mask = np.all(idx_pairs==idx_pair, axis=1)
#             idx_pair_frames = frames[mask]
#             idx_pair_counts = counts[mask]
#             all_frames = np.arange(min(idx_pair_frames), max(idx_pair_frames)+1)
#             all_counts = np.full(len(all_frames), np.nan)
#             all_counts[np.searchsorted(all_frames, idx_pair_frames)] = idx_pair_counts
#             phago_ds = phago_group.create_dataset(f'{idx_pair[0]:04}_{idx_pair[1]:04}', shape=(len(all_frames), 2), maxshape=(None, None), dtype=np.float32)
#             phago_ds.attrs['columns'] = ['frame', 'number_of_pathogen_pixels']
#             phago_ds[:, 0] = all_frames
#             phago_ds[:, 1] = all_counts


def phagocytosis(epi_mask, phase_mask, start_frame):
    overlap = (epi_mask>0) & (phase_mask>0)
    epi_idxs = epi_mask[overlap]
    phase_idxs = phase_mask[overlap]
    frames = np.nonzero(overlap)[0] + start_frame
    events, counts = np.unique(np.stack((phase_idxs, epi_idxs, frames), axis=1), axis=0, return_counts=True)
    idx_pairs = events[:, :2]
    frames = events[:, 2]
    phago_events = {}
    for idx_pair in np.unique(events[:, :2], axis=0):
        mask = np.all(idx_pairs==idx_pair, axis=1)
        idx_pair_frames = frames[mask]
        idx_pair_counts = counts[mask]
        all_frames = np.arange(min(idx_pair_frames), max(idx_pair_frames)+1)
        all_counts = np.full(len(all_frames), np.nan)
        all_counts[np.searchsorted(all_frames, idx_pair_frames)] = idx_pair_counts
        phago_events[f'{idx_pair[0]:04}_{idx_pair[1]:04}'] = np.stack((all_frames, all_counts), axis=1)
    return phago_events

# def phagocytosis(epi_mask, phase_mask, start_frame):
#     '''
#     Returns:
#         phago_events []
#     '''
#     overlap = (epi_mask>0) & (phase_mask>0)
#     epi_idxs = epi_mask[overlap]
#     phase_idxs = phase_mask[overlap]
#     frames = np.nonzero(overlap)[0] + start_frame
#     events, counts = np.unique(np.stack((phase_idxs, epi_idxs, frames), axis=1), axis=0, return_counts=True)
#     idx_pairs = events[:, :2]
#     frames = events[:, 2]
#     phago_events = {}
#     for idx_pair in np.unique(events[:, :2], axis=0):
#         mask = np.all(idx_pairs==idx_pair, axis=1)
#         idx_pair_frames = frames[mask]
#         idx_pair_counts = counts[mask]
#         all_frames = np.arange(min(idx_pair_frames), max(idx_pair_frames)+1)
#         all_counts = np.full(len(all_frames), np.nan)
#         all_counts[np.searchsorted(all_frames, idx_pair_frames)] = idx_pair_counts
#         phago_events[f'{idx_pair[0]:04}_{idx_pair[1]:04}'] = np.stack((all_frames, all_counts), axis=1)
#     return phago_events

def intensity(expanded_phase_masks, phase_ims):
    intensities = expanded_phase_masks * phase_ims/255
    return torch.mean(intensities, dim=(1,2)).cpu(), torch.var(intensities, dim=(1,2)).cpu()

def perimeter(expanded_phase_masks):
    kernel = torch.tensor([[1, 1, 1],
                           [1, 9, 1],
                           [1, 1, 1]]).to(device)
    padded_masks = torch.nn.functional.pad(expanded_phase_masks, (1, 1, 1, 1), mode='constant', value=0)
    conv_result = torch.nn.functional.conv2d(padded_masks.unsqueeze(1).float(), kernel.unsqueeze(0).unsqueeze(0).float(),
                                                padding=0).squeeze()
    return torch.sum((conv_result >= 10) & (conv_result <=16), dim=(1, 2)).float()

def frame_by_frame_features(phase_masks, phase_ims, cell_batchsize, num_cells):
    intensity_means = []
    intensity_variances = []
    perimeters = []
    mode_scores = []
    phase_ims = torch.tensor(phase_ims)
    for i, (phase_mask, phase_im) in enumerate(zip(phase_masks, phase_ims)):
        meansi = torch.empty(size=(num_cells,))
        variancesi = torch.empty(size=(num_cells,))
        perimetersi = torch.empty(size=(num_cells,))
        phase_im = phase_im.to(device)
        for first_cell in range(0, num_cells, cell_batchsize):
            last_cell = min(first_cell+cell_batchsize, num_cells)
            cell_idxs = torch.arange(first_cell, last_cell).to(device)
            expanded_masks = torch.tensor(phase_mask).to(device).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)
            meansi[first_cell:last_cell], variancesi[first_cell:last_cell] = intensity(expanded_masks, phase_im)
            perimetersi[first_cell:last_cell] = perimeter(expanded_masks)
        intensity_means.append(meansi)
        intensity_variances.append(variancesi)
        perimeters.append(perimetersi)

        mode_scores.append(fitting.apply_pca_kmeans(i))
    return np.concatenate((np.stack((np.array(intensity_means), np.array(intensity_variances), np.array(perimeters)), axis=2), np.array(mode_scores)), axis=2)
    # return np.stack((np.array(intensity_means), np.array(intensity_variances), np.array(perimeters), np.array(mode_scores)), axis=2)

# def get_efds(phase_masks, num_cells, f, first_frame, last_frame, efd_order=10):
#     efds_list = []
#     for i, phase_mask in enumerate(phase_masks):
#         frame = first_frame + i
#         cell_contours = mask_funcs.get_border_representation(phase_mask, num_cells, f, frame)
#         efds = np.array([elliptic_fourier_descriptors(cell_contour, order=10).flatten()[3:] if len(cell_contour)>0 else np.full(efd_order*4 - 3, np.nan) for cell_contour in cell_contours])
#         efds_list.append(efds)
#     efds = np.array(efds_list)
#     return efds

def derived_features(f, first_frame, last_frame):
    # speed, displacement, displacement speed
    all_xs, all_ys = tools.get_features_ds(f['Cells']['Phase'], 'x'), tools.get_features_ds(f['Cells']['Phase'], 'y')
    x0, y0 = all_xs[~np.isnan(all_xs)][0], all_ys[~np.isnan(all_ys)][0]
    xs = all_xs[first_frame:last_frame]
    ys = all_ys[first_frame:last_frame]
    
    pathogen_xs = tools.get_features_ds(f['Cells']['Epi'], 'x')[first_frame:last_frame]
    pathogen_ys = tools.get_features_ds(f['Cells']['Epi'], 'y')[first_frame:last_frame]

    displacements = np.sqrt((xs-x0)**2 + (ys-y0)**2)
    # displacement_speeds = displacements / np.arange(first_frame, last_frame)[:, np.newaxis]
    displacement_speeds = np.diff(displacements, axis=0)

    phago_phago_distances = np.linalg.norm(np.expand_dims(np.stack((xs, ys), 2), 1) - np.expand_dims(np.stack((xs, ys), 2), 2), axis=3)
    phago_pathogen_distances = np.linalg.norm(np.expand_dims(np.stack((xs, ys), 2), 1) - np.expand_dims(np.stack((pathogen_xs, pathogen_ys), 2), 2), axis=3)

    phago_densities = np.sum(phago_phago_distances < 150, axis=1) - 1
    pathogen_densities = np.sum(phago_pathogen_distances < 150, axis=1)

    print(phago_phago_distances.shape, phago_pathogen_distances.shape, phago_densities.shape, pathogen_densities.shape)
    if first_frame != 0: 
        xs = np.insert(xs, 0, tools.get_features_ds(f['Cells']['Phase'], 'x')[first_frame-1], axis=0)
        ys = np.insert(ys, 0, tools.get_features_ds(f['Cells']['Phase'], 'y')[first_frame-1], axis=0)
    
    xspeeds = np.diff(xs, axis=0)
    yspeeds = np.diff(ys, axis=0)
    speeds = np.sqrt(xspeeds**2 + yspeeds**2)

    if first_frame == 0: 
        speeds = np.insert(speeds, 0, np.full(xs.shape[1], np.nan), axis=0)
        displacement_speeds = np.insert(displacement_speeds, 0, np.full(xs.shape[1], np.nan), axis=0)

    return np.stack((speeds, displacements, displacement_speeds, phago_densities, pathogen_densities), axis=2)


def get_features(file=SETTINGS.DATASET, frame_batchsize=25, cell_batchsize=50):
    with h5py.File(file, 'r+') as f:
        # First need to fit pca and kmeans clustering models
        print('Fitting PCA and KMeans clustering models')
        fitting.fit_pca_kmeans()

        print('\nExtracting Features')
        # efd_order = 10
        phago_group = f.create_group('Cells/Phagocytosis')
        num_frames = len(list(f['Images']['Phase'].keys()))
        num_cells = f['Cells']['Phase'][:].shape[1]

        # efd_names = [f'efd{i}{j:04}' for i in range(4) for j in range(efd_order)][3:]
        mode_names = [f'mode{i}' for i in range(SETTINGS.KMEANS_CLUSTERS)]
        # modes = np.arange(SETTINGS.KMEANS_CLUSTERS).astype(str)
        frame_by_frame_feature_names = np.append(['intensity_mean', 'intensity_variance', 'perimeter'], mode_names)
        derived_feature_names = ['speed', 'displacement', 'displacement_speed', 'phagocyte density', 'pathogen_density']
        all_new_feature_names = np.append(frame_by_frame_feature_names, derived_feature_names)
        f['Cells']['Phase'].attrs['features'] = np.append(f['Cells']['Phase'].attrs['features'], all_new_feature_names)


        f['Cells']['Phase'].resize(len(list(f['Cells']['Phase'].attrs['features'])), axis=2)
        for first_frame in range(0, num_frames, frame_batchsize):
            last_frame = min(first_frame + frame_batchsize, num_frames)

            sys.stdout.write(f'\rFrames {first_frame} to {last_frame} / {SETTINGS.NUM_FRAMES}')
            sys.stdout.flush()

            epi_mask = tools.get_masks(f, 'Epi', first_frame=first_frame, last_frame=last_frame)
            phase_mask = tools.get_masks(f, 'Phase', first_frame=first_frame, last_frame=last_frame)

            epi_im = tools.get_images(f, 'Epi', first_frame=first_frame, last_frame=last_frame)
            phase_im = tools.get_images(f, 'Phase', first_frame=first_frame, last_frame=last_frame)

            #get_efds(phase_masks=phase_mask, num_cells=num_cells, f=f, first_frame=first_frame, last_frame=last_frame)
            #intensity_np(phase_mask, phase_im)
            f['Cells']['Phase'][first_frame:last_frame, :, 3:] = np.concatenate((frame_by_frame_features(phase_mask, phase_im, cell_batchsize, num_cells), 
                                                                                 derived_features(f, first_frame, last_frame)), 
                                                                                 axis=2)

            phago_events = phagocytosis(epi_mask, phase_mask, first_frame)
            for phago_event_name, phago_event_data in phago_events.items():
                if phago_event_name not in phago_group.keys():    
                    phago_ds = phago_group.create_dataset(phago_event_name, data=phago_event_data, maxshape=(None, None))
                    phago_ds.attrs['phagocyte_idx'] = phago_event_name[:4]
                    phago_ds.attrs['pathogen_idx'] = phago_event_name[-4:]
                else:
                    # pad top of new array with np.nan count value if gap in frames from end of last array
                    padding = np.array([[float(frame), np.nan] for frame in range(int(phago_group[phago_event_name][-1, 0])+1, int(phago_event_data[0, 0]))])
                    if len(padding) > 0:
                        phago_event_data = np.concatenate((padding, phago_event_data), axis=0)
                    phago_group[phago_event_name].resize((len(phago_group[phago_event_name]) + len(phago_event_data), 2))
                    phago_group[phago_event_name][-len(phago_event_data):] = phago_event_data

        nan_time_cells = np.nonzero(np.isnan(tools.get_features_ds(f['Cells']['Phase'], 'x')))
        ds = f['Cells']['Phase'][:]
        ds[nan_time_cells] = np.nan
        f['Cells']['Phase'][()] = ds

def main():
    get_features()

if __name__ == '__main__':
    main()