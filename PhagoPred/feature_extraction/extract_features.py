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

        if self.derived_feature:
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
    
    # def get_feature_names(self):

    #     for feature in self.primary_features + self.derived_features:
    #         for name in feature.get_names():
    #             self.feature_names.append(name)
    #     return self.feature_names
    
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
        current_idx = 0
        for feature_type in (self.primary_features, self.derived_features):
            for feature in feature_type:
                feature.set_index_positions(current_idx, current_idx+len(feature.get_names()))
                current_idx += 1

    def get_features_xr(self, h5py_file) -> xr.Dataset:
        """
        h5py file is open file, returns x_array (chunked like h5py)
        """
        print(self.features_ds)
        return xr.open_dataset(h5py_file, group=self.features_ds, chunks='auto', engine="h5netcdf")
    
class FeaturesExtraction:
    FRAME_DIM = 0
    CELL_DIM = 1
    FEATURES_DIM = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(
            self, 
            h5py_file=SETTINGS.DATASET, 
            frame_batchsize=500,
            cell_batch_size=100):
        
        self.num_frames = None
        
        self.h5py_file = h5py_file
        self.frame_batchsize = frame_batchsize
        self.cell_batch_size = cell_batch_size

        self.morphology_model = MorphologyFit(self.h5py_file)

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
        Fit any models which need fitting
        Prepare hdf5 file
        """
        print("***FITTING MORPHOLOGY MODEL***")
        # self.morphology_model.fit()

        with h5py.File(self.h5py_file, 'r+') as f:
            self.num_frames = f[self.cell_types[0].features_ds].shape[self.FRAME_DIM]
            for cell_type in self.cell_types:

                dataset = f[cell_type.features_ds]

                cell_type.set_initial_num_features(dataset.shape[self.FEATURES_DIM])
            
                features_list = cell_type.get_feature_names()
                dataset.attrs['features'] = features_list

                dataset.attrs['dimensions'] = ['frames', 'cells', 'features']

                dataset.resize(cell_type.initial_num_features + len(features_list), axis=2)

                cell_type.set_num_cells(dataset.shape[self.CELL_DIM])

    def extract_primary_features(self, f: h5py.File, cell_type: CellType) -> None:
        for frame_idx in range(self.num_frames):

            frame_results = np.full((cell_type.num_cells, len(cell_type.primary_feature_names)), np.nan)

            mask = cell_type.get_masks(f, frame_idx)
            image = cell_type.get_images(f, frame_idx)

            for first_cell in range(0, cell_type.num_cells, self.cell_batch_size):

                last_cell = min(first_cell + self.cell_batch_size, cell_type.num_cells)

                cell_idxs = torch.tensor.arange(first_cell, last_cell).to(self.DEVICE)

                expanded_mask = torch.tensor(mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                for feature in cell_type.primary_features:
                    frame_results[first_cell:last_cell, feature.index_positions[0]:feature.index_positions[1]]

            f[cell_type.feature_ds][frame_idx, :, 0:len(cell_type.primary_feature_names)] = frame_results

    def extract_derived_features(self, f: h5py.File, cell_type: CellType) -> None:
        phase_features_xr = self.cell_types[self.cell_type_names.index('Phase')].get_features_xr(f)
        # epi_features_xr = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_xr(f)

        print(phase_features_xr)

        # for first_frame in range(0, self.num_frames, self.frame_batchsize):
        #     last_frame = min(first_frame + self.frame_batchsize, self.num_frames)

        #     batch_results = np.full((last_frame-first_frame, cell_type.num_cells, len(cell_type.derived_feature_names)), np.nan)

        #     if first_frame != 0:
        #         last_frame += 1

        #     np_slice = np.s_[first_frame:last_frame, :, :]
                
    # def get_slice(self, group: str, np_slice: np.slice) -> pd.DataFrame:
    #     data = ds[np_slice]

        # xr_dset = xr.open_dataset(self.)
        # for dim in data.ndim:

            

    
    # def extract_features(self):
    #     with h5py.File(self.h5py_file, 'r+') as f:
    #         for first_frame in range(0, self.num_frames, self.frame_batchsize):
    #             last_frame = min(first_frame + self.frame_batchsize, self.num_frames)

    #             sys.stdout.write(f'\rFrames {first_frame} to {last_frame} / {self.num_frames}')
    #             sys.stdout.flush()

    #             for cell_type in self.cell_types:

    #                 batch_results = np.full((len(last_frame-first_frame), cell_type.num_cells, len(cell_type.feature_names)), np.nan)

    #                 if len(cell_type.primary_features) > 0:
    #                     masks = cell_type.get_masks(first_frame, last_frame, f)
    #                     images = cell_type.get_images(first_frame, last_frame, f)

    #                     for frame_idx, (frame_mask, frame_image) in enumerate(zip(masks, images)):

    #                         for first_cell in range(0, self.num_cells, self.cell_batch_size):
    #                             last_cell = min(first_cell + self.cell_batch_size, self.num_cells)

    #                             cell_idxs = torch.arange(first_cell, last_cell).to(self.DEVICE)
    #                             expanded_mask = torch.tensor(frame_mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

    #                             for feature in cell_type.primary_features:
    #                                 result = feature.compute(expanded_mask, frame_image)
                                    
    #                                 batch_results[frame_idx, 
    #                                               first_cell:last_cell, 
    #                                               feature.get_index_positions()[0]:feature.get_index_poitions[-1]
    #                                               ] = result
    #                                 # NEED TO WRITE PRIMARY FEATURES TO H5PY BEFORE CALCULATING SECONDARY
                                    
    #                 if len(cell_type.derived_features) > 0:
    #                     phase_features_dataset = self.cell_types[np.argwhere(self.cell_type_names == 'Phase')].get_features_ds()
    #                     epi_features_dataset = self.cell_types[np.argwhere(self.cell_type_names == 'Epi')].get_features_ds()
    #                     for feature in cell_type.derived_features:
    #                         result = feature.compute(phase_features_dataset, epi_features_dataset)
    #                         batch_results[
    #                             first_frame:last_frame,
    #                             :,
    #                             feature.get_index_positions()[0], feature.get_index_positions()[1]
    #                         ] = result

def main():
    feature_extractor = FeaturesExtraction()
    # feature_extractor.set_up()

    # feature_extractor.extract_derived_features()
    with h5py.File('PhagoPred/Datasets/mac_07_03_short.h5', 'r') as f:
        # print(f['Cells/Phase'].encoding)
        feature_extractor.extract_derived_features(f, 'Phase')

if __name__ == '__main__':
    main()
        