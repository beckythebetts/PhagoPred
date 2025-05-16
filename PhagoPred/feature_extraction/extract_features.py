import sys

import h5py
import numpy as np
import torch

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction import features
from PhagoPred.utils import tools

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

        self.cell_types = [CellType('Phase'), CellType('Epi')]
        self.cell_type_names = [cell_type.name for cell_type in self.cell_types]

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
        self.morphology_model.fit()

        with h5py.File(self.h5py_file, 'r+') as f:
            self.num_frames = f[self.cell_types[0].features_ds].shape[self.FRAME_DIM]
            for cell_type in self.cell_types:

                dataset = f[cell_type.features_ds]
            
                features_list = cell_type.get_feature_names()
                dataset.attrs['features'] = features_list

                dataset.attrs['dimensions'] = ['frames', 'cells', 'features']

                dataset.resize(len(features_list), axis=2)

                cell_type.set_num_cells(dataset.shape[self.CELL_DIM])

    def extract_features(self):
        with h5py.File(self.h5py_file, 'r+') as f:
            for first_frame in range(0, self.num_frames, self.frame_batchsize):
                last_frame = min(first_frame + self.frame_batchsize, self.num_frames)

                sys.stdout.write(f'\rFrames {first_frame} to {last_frame} / {self.num_frames}')
                sys.stdout.flush()

                for cell_type in self.cell_types:

                    batch_results = np.full((len(last_frame-first_frame), cell_type.num_cells, len(cell_type.feature_names)), np.nan)

                    if len(cell_type.primary_features) > 0:
                        masks = cell_type.get_masks(first_frame, last_frame, f)
                        images = cell_type.get_images(first_frame, last_frame, f)

                        for frame_idx, (frame_mask, frame_image) in enumerate(zip(masks, images)):

                            for first_cell in range(0, self.num_cells, self.cell_batch_size):
                                last_cell = min(first_cell + self.cell_batch_size, self.num_cells)

                                cell_idxs = torch.arange(first_cell, last_cell).to(self.DEVICE)
                                expanded_mask = torch.tensor(frame_mask).to(self.DEVICE).unsqueeze(0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                                primary_batch_features = []

                                for feature in cell_type.primary_features:
                                    result = feature.compute(expanded_mask, frame_image)

                                    if result.ndim==1:
                                        result = result[:, ]

                                    

                            batch_results[frame_idx, first_cell:last_cell, :len(primary_batch_features)]

class CellType:

    def __init__(self, name):
        self.features_ds = f'Cells/{name}'
        self.images = f'Images/{name}'
        self.masks = f'Segmentations/{name}'

        self.primary_features = []
        self.derived_features = []
        
        self.feature_names = []
        self.primary_feature_names = []
        self.derived_feature_names = []

        self.num_cells = None

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
    
    def get_feature_names(self):

        for feature in self.primary_features + self.derived_features:
            for name in feature.get_names():
                self.feature_names.append(name)
        return self.feature_names
    
    def get_masks(self, first_frame: int, last_frame: int, h5py_file: h5py.File) -> np.array:
        """
        Gets masks between the first and last frame, assuming h5py file is open and readable.
        masks dims = [num_frames, x, y]
        """
        frame_list = list(h5py_file[self.masks].keys())[first_frame:last_frame]
        return np.array([h5py_file[self.masks][frame][:] for frame in frame_list], dtype='int16')
    
    def get_images(self, first_frame: int, last_frame: int, h5py_file: h5py.File) -> np.array:
        """
        Gets images between the first and last frame, assuming h5py file is open and readable.
        dims = [num_frames, x, y]
        """
        frame_list = list(h5py_file[self.images].keys())[first_frame:last_frame]
        return np.array([h5py_file[self.images][frame][:] for frame in frame_list], dtype='uint8')

    def set_num_cells(self, num_cells):
        self.num_cells = num_cells
    
    def set_feature_idxs(self):
        current_idx = 0
        for feature_type in (self.primary_features, self.derived_features):
            for feature in feature_type:
                feature.set_index_positions(current_idx, current_idx+len(feature.get_names()))

def main():
    feature_extractor = FeaturesExtraction()
        