import sys

import h5py
import numpy as np
import torch
import pandas as pd
import xarray as xr

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction import features
from PhagoPred.utils import tools


class CellType:
    FRAME_DIM = 0
    CELL_DIM = 1
    FEATURES_DIM = 2

    DIMS = ["Frame", "Cell Index", "Feature"]

    def __init__(self, name: str) -> None:

        self.name = name

        self.features_ds = f'Cells/{name}'
        self.images = f'Images/{name}'
        self.masks = f'Segmentations/{name}'

        self.primary_features = []
        self.derived_features = []
        
        self.feature_names = []
        self.primary_feature_names = []
        self.derived_feature_names = []

        self.initial_num_features = None

        self.num_cells = None

    def set_initial_num_features(self, num: int) -> None:
        self.initial_num_features = num

    def add_feature(self, feature):
        if feature.primary_feature:
            self.primary_features.append(feature)

        if feature.derived_feature:
            self.derived_features.append(feature)

    def get_feature_names(self):

        for feature in self.primary_features:
            for name in feature.get_names():
                self.primary_feature_names.append(name)

        for feature in self.derived_features:
            for name in feature.get_names():
                self.derived_feature_names.append(name)

        self.feature_names = self.primary_feature_names + self.derived_feature_names
        return self.feature_names
    
    def get_masks(self, h5py_file: h5py.File, first_frame: int, last_frame: int = None) -> np.array:
        """
        Gets masks between the first and last frame, assuming h5py file is open and readable.
        masks dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1
        frame_list = list(h5py_file[self.masks].keys())[first_frame:last_frame]
        masks = np.array([h5py_file[self.masks][frame][:] for frame in frame_list], dtype='int16')
        if len(masks) == 1:
            return masks[0]
        else:
            return masks
    
    def get_images(self, h5py_file: h5py.File, first_frame: int, last_frame: int = None) -> np.array:
        """
        Gets images between the first and last frame, assuming h5py file is open and readable.
        dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1
        frame_list = list(h5py_file[self.images].keys())[first_frame:last_frame]
        images = np.array([h5py_file[self.images][frame][:] for frame in frame_list], dtype='uint8')
        if len(images) == 1:
            return images[0]
        else:
            return images

    def set_num_cells(self, num_cells):
        self.num_cells = num_cells
    
    def set_feature_idxs(self):
        for feature_type in (self.primary_features, self.derived_features):
            current_idx = 0
            for feature in feature_type:
                feature_len = len(feature.get_names())
                feature.set_index_positions(current_idx, current_idx+feature_len)
                current_idx += feature_len

    # def set_feature_idxs(self):
    # #     current_idx = 0
    #     for feature 

    def set_up_features_ds(self, h5py_file: h5py.File):
        dataset = h5py_file[self.features_ds]
        dataset.attrs['dimensions'] = self.DIMS

        self.set_initial_num_features(dataset.shape[self.DIMS.index("Feature")])
    
        features_list = self.get_feature_names()
        if len(features_list) > 0:
            # for feature_name in features_list:
            #     if feature_name not in dataset.attrs['features']:
            #         dataset.attrs['features'] = np.append(dataset.attrs['features'], feature_name)
            dataset.attrs['features'] = np.concatenate((dataset.attrs['features'], np.array(features_list)))

        dataset.attrs['dimensions'] = ['frames', 'cells', 'features']

        dataset.resize(self.initial_num_features + len(features_list), axis=2)

        self.set_num_cells(dataset.shape[self.DIMS.index("Cell Index")])

    def get_features_xr(self, h5py_file: h5py.File) -> xr.Dataset:
        """
        h5py file is open file, returns x_array (chunked like h5py)
        """
        data = h5py_file[self.features_ds]
        feature_names = data.attrs['features']
        return xr.DataArray(data, dims=self.DIMS, coords={"Feature":feature_names})


    
class FeaturesExtraction:
 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self, 
            h5py_file=SETTINGS.DATASET, 
            frame_batchsize=10,
            cell_batch_size=100):
        
        self.num_frames = None
        
        self.h5py_file = h5py_file
        self.frame_batchsize = frame_batchsize
        self.cell_batch_size = cell_batch_size

        self.cell_types = [CellType('Phase')]
        self.cell_type_names = [cell_type.name for cell_type in self.cell_types]

        self.set_up()

    def add_feature(self, feature, cell_type: str) -> None:
        """
        Add feature to specified cell type
        """
        self.cell_types[self.cell_type_names.index(cell_type)].add_feature(feature)

    def set_up(self):
        """
        Prepare hdf5 file
        """

        with h5py.File(self.h5py_file, 'r+') as f:
            self.num_frames = f[self.cell_types[0].features_ds].shape[self.cell_types[0].DIMS.index("Frame")]

            for cell_type in self.cell_types:
                cell_type.set_num_cells(f[cell_type.features_ds].shape[cell_type.DIMS.index("Cell Index")])
                cell_type.set_feature_idxs()


                cell_type.set_initial_num_features(f[cell_type.features_ds].shape[cell_type.DIMS.index("Feature")])

                # cell_type.set_up_features_ds(f)

                cell_type.set_initial_num_features(cell_type.initial_num_features-1)
                cell_type.get_feature_names()

    def extract_features(self) -> None:
        with h5py.File(self.h5py_file, 'r+') as f:
            for cell_type in self.cell_types:
                self.extract_primary_features(f, cell_type)
                self.extract_derived_features(f, cell_type)

    def extract_primary_features(self, f: h5py.File, cell_type: CellType) -> None:
        if len(cell_type.primary_features) > 0:
            print(f'\nCalculating primary features for {cell_type.name}\n')
            for frame_idx in range(self.num_frames):

                sys.stdout.write(f'\rFrame {frame_idx+1} / {self.num_frames}')
                sys.stdout.flush()

                frame_results = np.full((cell_type.num_cells, len(cell_type.primary_feature_names)), np.nan)

                mask = cell_type.get_masks(f, frame_idx)
                image = cell_type.get_images(f, frame_idx)

                for first_cell in range(0, cell_type.num_cells, self.cell_batch_size):

                    last_cell = min(first_cell + self.cell_batch_size, cell_type.num_cells)

                    cell_idxs = torch.arange(first_cell, last_cell).to(self.DEVICE)

                    expanded_mask = torch.tensor(mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                    if first_cell == 0:
                        expanded_mask[0] = torch.zeros_like(expanded_mask[0])
                    for feature in cell_type.primary_features:
                        result = feature.compute(mask=expanded_mask, image=image)
                        if result.ndim == 1:
                            result = result[:, np.newaxis]
                        frame_results[first_cell:last_cell, feature.index_positions[0]:feature.index_positions[1]] = result

                f[cell_type.features_ds][frame_idx, :, cell_type.initial_num_features:cell_type.initial_num_features+len(cell_type.primary_feature_names)] = frame_results

    def extract_derived_features(self, f: h5py.File, cell_type: CellType) -> None:
        if len(cell_type.derived_features) > 0:
            print(f'\nCalculating derived features for {cell_type.name}\n')

            phase_features_xr = self.cell_types[self.cell_type_names.index('Phase')].get_features_xr(f)

            phase_features_xr = phase_features_xr.chunk({'Frame': self.frame_batchsize, 'Cell Index': self.cell_batch_size})

            epi_features_xr = None
            if 'Epi' in self.cell_type_names:
                epi_features_xr = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_xr(f)
                epi_features_xr = epi_features_xr.chunk({'Frame': self.frame_batchsize, 'Cell index': self.cell_batch_size})

            for feature in cell_type.derived_features:
                result = feature.compute(phase_xr=phase_features_xr, epi_xr=epi_features_xr)

                first_idx = -len(cell_type.derived_feature_names)+feature.index_positions[0]
                last_idx = -len(cell_type.derived_feature_names)+feature.index_positions[1]
                if last_idx == 0:
                    last_idx = None

                f[cell_type.features_ds][:, :, first_idx:last_idx] = result

def main():
    feature_extractor = FeaturesExtraction()

    phase_features = [
        # features.MorphologyModes(), 
        # features.Speed(),
        # features.DensityPhase(),
        # features.Displacement()
        # features.Perimeter(),
        # features.Circularity()
        features.GaborScale()
        ]
    
    for feature in phase_features:
        feature_extractor.add_feature(feature, 'Phase')

    feature_extractor.set_up()
    feature_extractor.extract_features()

if __name__ == '__main__':
    main()
        