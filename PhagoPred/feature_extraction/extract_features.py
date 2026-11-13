from __future__ import annotations
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import pandas as pd
import xarray as xr
import dask.array as da
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit
from PhagoPred.feature_extraction import features
from PhagoPred.utils.h5_file import H5File
from PhagoPred.utils.logger import get_logger

log = get_logger()


class CellType:
    FRAME_DIM = 0
    CELL_DIM = 1

    DIMS = ["Frame", "Cell Index"]

    def __init__(self, name: str) -> None:

        self.name = name

        self.features_group = f'Cells/{name}'
        self.images = f'Images/{name}'
        self.masks = f'Segmentations/{name}'

        self.primary_features = []
        self.derived_features = []
        self.primary_derived_features = []

        self.feature_names = []

        self.primary_feature_names = []
        self.derived_feature_names = []
        self.primary_derived_feature_names = []

        self.primary_feature_datasets = []
        self.derived_feature_datasets = []
        self.primary_derived_feature_datasets = []

        self.num_cells = None

    def set_start_end_death_frames(self):
        """Add atributes  to self.features_group consisting of lists of frames of first appearnance, last appearance and death for each cell.
        For no cell death observecm dell death will be -1."""

    def add_feature(self, feature):
        if feature.primary_feature:
            self.primary_features.append(feature)
        else:
            if isinstance(feature, features.CellDeath):
                self.derived_features.insert(0, feature)
            else:
                self.derived_features.append(feature)

    def get_feature_names(self):

        for feature in self.primary_features:
            for name in feature.get_names():
                self.primary_feature_names.append(name)

        for feature in self.derived_features:
            for name in feature.get_names():
                self.derived_feature_names.append(name)

        for feature in self.primary_derived_features:
            for name in feature.get_names():
                self.primary_derived_feature_names.append(name)

        self.feature_names = self.primary_feature_names + self.derived_feature_names + self.primary_derived_feature_names
        return self.feature_names

    def get_masks(self,
                  h5py_file: h5py.File,
                  first_frame: int,
                  last_frame: int = None) -> np.array:
        """
        Gets masks between the first and last frame, assuming h5py file is open and readable.
        masks dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1

        masks = h5py_file[self.masks][first_frame:last_frame]
        if len(masks) == 1:
            return masks[0]
        else:
            return masks

    def get_images(self,
                   h5py_file: h5py.File,
                   first_frame: int,
                   last_frame: int = None) -> np.array:
        """
        Gets images between the first and last frame, assuming h5py file is open and readable.
        dims = [num_frames, x, y]
        """
        if last_frame is None:
            last_frame = first_frame + 1
        try:
            images = h5py_file[self.images][first_frame:last_frame]
        except KeyError:
            images = np.zeros(
                (last_frame - first_frame,
                 h5py_file['Images'].attrs['Image size / pixels'][0],
                 h5py_file['Images'].attrs['Image size / pixels'][1]))

        if len(images) == 1:
            return images[0]

        else:
            return images

    def set_num_cells(self, num_cells):
        self.num_cells = num_cells

    def set_up_features_group(self, h5py_file: h5py.File):
        """
        Create hdf5 datset for each feature
        """
        num_frames = h5py_file[self.images].shape[self.FRAME_DIM]

        num_cells = 1

        if 'X' in h5py_file[self.features_group].keys():
            num_cells = h5py_file[self.features_group]['X'].shape[1]
        for feature_name in self.primary_feature_names:
            dataset = h5py_file.require_dataset(
                f'{self.features_group}/{feature_name}',
                shape=(num_frames, num_cells),
                maxshape=(num_frames, None),
                dtype=np.float32,
                fillvalue=np.nan,
                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.primary_feature_datasets.append(dataset)

        for feature_name in self.derived_feature_names:
            dataset = h5py_file.require_dataset(
                f'{self.features_group}/{feature_name}',
                shape=(num_frames, num_cells),
                maxshape=(num_frames, None),
                dtype=np.float32,
                fillvalue=np.nan,
                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.derived_feature_datasets.append(dataset)

        for feature_name in self.primary_derived_feature_names:
            dataset = h5py_file.require_dataset(
                f'{self.features_group}/{feature_name}',
                shape=(num_frames, num_cells),
                maxshape=(num_frames, None),
                dtype=np.float32,
                fillvalue=np.nan,
                exact=True)
            dataset.attrs['dimensions'] = self.DIMS
            self.primary_derived_feature_datasets.append(dataset)

    def get_features_xr(self,
                        h5py_file: h5py.File,
                        features: list = None) -> xr.Dataset:
        """
        h5py file is open file, returns x_array (chunked like h5py). If features not specified, reads all features iwth both cell index and time dimensions.
        """
        data_dict = {}
        shape = h5py_file[self.features_group]['Area'].shape
        for feature_name, feature_data in h5py_file[
                self.features_group].items():
            if features:
                if feature_name not in features:
                    continue
            data = da.from_array(feature_data, chunks=feature_data.chunks)

            if feature_data.shape[0] == 1:
                data = da.broadcast_to(data, shape)

            data_dict[feature_name] = (self.DIMS, data)

        ds = xr.Dataset(data_dict)

        return ds


class FeaturesExtraction:

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIMS = ['Frame', 'Cell Index']

    def __init__(
        self,
        feature_list: list[features.BaseFeature],
        h5py_file=SETTINGS.DATASET,
        frame_batchsize=10,
        cell_batch_size=100,
    ) -> None:
        self.h5py_file = H5File(h5py_file)
        self.frame_batchsize = frame_batchsize
        self.cell_batch_size = cell_batch_size

        self.primary_features = [f for f in feature_list if f.primary_feature]
        self.derived_features = [f for f in feature_list if f.derived_feature]

    def extract_features(self) -> None:

        self.extract_primary_features()
        self.extract_derived_features()
        self.h5py_file.write_attr('Epi threshold', H5File.EPI_THRESHOLD)

    def extract_primary_features(self) -> None:
        if len(self.primary_features) > 0:
            print(f'\n=== Calculating Primary Features ===\n')
            num_frames, num_cells = self.h5py_file.num_frames, self.h5py_file.num_cells
            results = {}
            crop_features = [
                feature for feature in self.primary_features if feature.crop
            ]
            no_crop_features = [
                feature for feature in self.primary_features
                if not feature.crop
            ]

            if len(crop_features) > 0:
                centre_coords = self.h5py_file.get_features(['X', 'Y'],
                                                            filter_bad=False)
                x_centres = torch.from_numpy(centre_coords['X'])
                y_centres = torch.from_numpy(centre_coords['Y'])

                x_centres, y_centres = x_centres.to(self.DEVICE), y_centres.to(
                    self.DEVICE)

            for frame_idx in tqdm(range(num_frames), unit='frames'):

                mask = self.h5py_file.get_segs(frame_idx)
                mask = torch.tensor(mask).to(self.DEVICE)
                binary_mask = mask >= 0

                # image = torch.tensor(self.h5py_file.get_phase(frame_idx)).to(
                #     self.DEVICE)
                image = None
                epi_image = torch.tensor(
                    self.h5py_file.get_epi(frame_idx, threshold=True).astype(
                        np.float32)).to(self.DEVICE)

                for crop, features_list in zip(
                    [False, True], [no_crop_features, crop_features]):
                    if len(features_list) > 0:

                        for first_cell in range(0, num_cells,
                                                self.cell_batch_size):

                            last_cell = min(first_cell + self.cell_batch_size,
                                            num_cells)

                            cell_idxs = torch.arange(first_cell,
                                                     last_cell).to(self.DEVICE)

                            expanded_mask = mask.unsqueeze(
                                0) == cell_idxs.unsqueeze(1).unsqueeze(2)

                            if crop:
                                current_x_centres, current_y_centres = x_centres[
                                    frame_idx,
                                    first_cell:last_cell], y_centres[
                                        frame_idx, first_cell:last_cell]
                                current_expanded_mask, current_epi_images, current_images, current_binary_masks = features.crop_masks_images(
                                    expanded_mask, epi_image, image,
                                    binary_mask, current_x_centres,
                                    current_y_centres)
                            else:
                                current_expanded_mask, current_epi_images, current_images, current_binary_masks = expanded_mask, epi_image, image, binary_mask

                            for feature in features_list:
                                result = feature.compute(
                                    mask=current_expanded_mask,
                                    image=current_images,
                                    epi_image=current_epi_images,
                                    binary_mask=current_binary_masks)
                                if result.ndim == 1:
                                    result = result[:, np.newaxis]
                                for i, feature_name in enumerate(
                                        feature.get_names()):
                                    if feature_name not in results:
                                        results[feature_name] = np.full(
                                            (num_frames, num_cells), np.nan,
                                            np.float32)
                                    results[feature_name][
                                        frame_idx,
                                        first_cell:last_cell] = result[:, i]
                            torch.cuda.empty_cache()

            self.h5py_file.write_features(results)

    def extract_derived_features(self) -> None:
        if len(self.derived_features) > 0:
            print('\n=== Calculating derived features ===\n')

            phase_features_xr = self._get_features_xr()
            shape = (self.h5py_file.num_frames, self.h5py_file.num_cells)
            results = {}
            for feature in self.derived_features:
                print(f'Calculating {", ".join(feature.get_names())}...')
                result = feature.compute(phase_xr=phase_features_xr,
                                         epi_xr=None)

                if result.ndim == 2:
                    result = result[:, :, np.newaxis]

                for i, feature_name in enumerate(feature.get_names()):
                    results[feature_name] = result[:, :, i]
                    # Later derived features can use this one (e.g. Death
                    # Frame), as when they were read lazily from the file
                    phase_features_xr[feature_name] = (self.DIMS,
                                                       np.broadcast_to(
                                                           result[:, :, i],
                                                           shape))

            self.h5py_file.write_features(results)

    def _get_features_xr(self) -> xr.Dataset:
        """Every feature in the file, in memory, as (Frame, Cell Index);
        per-cell features (one row) are broadcast over frames. Bad tracks
        aren't filtered."""
        with h5py.File(self.h5py_file.path, 'r') as f:
            names = list(f[H5File.FEATURES])
        features = self.h5py_file.get_features(names, filter_bad=False)
        shape = (self.h5py_file.num_frames, self.h5py_file.num_cells)
        ds = xr.Dataset({
            name: (self.DIMS, np.broadcast_to(values, shape))
            for name, values in features.items()
        })
        ds.attrs['im_shape'] = self.h5py_file.im_dims
        return ds


def extract_features(
    dataset=SETTINGS.DATASET,
    feature_list: features.BaseFeature | list[features.BaseFeature] = [
        # features.Fluorescence(),
        # features.ExternalFLuorcence(),
        features.Speed(),
        features.DensityPhase(),
        features.Displacement(),
        features.Perimeter(),
        features.Circularity(),
        features.Skeleton(),
        features.RegionProps(),
        # features.CellDeath(),
        features.FirstLastFrame(),
        features.EventDetection(),
        # features.EventDetection
        # features.ExternalFluorescence(),
        # features.Fluorescence(),
    ]):
    if isinstance(feature_list, features.BaseFeature):
        feature_list = [feature_list]
    FeaturesExtraction(feature_list=feature_list,
                       h5py_file=dataset).extract_features()


def main():
    datasets = ['16_10']
    # datasets_dir = Path('~/thor_server').expanduser() / 'MacrophageData'
    datasets_dir = Path('~/MacropjageData').expanduser()

    for dataset in datasets:
        dataset = datasets_dir / dataset
        for h5_file in dataset.glob('*.h5'):
            print(f'Extracting features for file {h5_file.name}')
            log.info(f'Extracting features for file at {h5_file}')
            extract_features(h5_file, features.RegionProps())


if __name__ == '__main__':
    main()
