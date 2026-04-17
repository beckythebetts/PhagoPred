from __future__ import annotations
from pathlib import Path

import numpy as np
import h5py
import pandas as pd
import xarray as xr


def load_h5(hdf5_paths: Path | list[Path], features: list[str] = None):
    if isinstance(hdf5_paths, Path):
        hdf5_paths = [hdf5_paths]
    all_data = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, 'r') as f:

            data = f['Cells']['Phase']
            if features is None:
                features = [
                    feat for feat in data.keys() if feat not in (
                        'Images',
                        'First Frame',
                        'Last Frame',
                        'X',
                        'Y',
                        'CellDeath',
                        'Macrophage',
                        'Dead Macrophage',
                        'Dead Phagocytes within 100 pixels',
                        'Dead Phagocytes within 250 pixels',
                        'Dead Phagocytes within 500 pixels',
                        'Alive Phagocytes within 100 pixels',
                        'Confidence Score',
                    )
                ]
            data = np.array([data[feat] for feat in features
                             ])  # (featues, frame, samples)

            # Set Speed to 1 for first frame to prevent NaN exclusion
            if 'Speed' in features:
                speed_idx = features.index('Speed')
                data[speed_idx, 0, :] = 1.0

            nan_mask = np.any(np.isnan(data), axis=(0, 1))
            # print(nan_mask, np.unique(nan_mask))
            data = data[:, :, ~nan_mask]
            data = data.transpose(2, 1, 0)  # (samples, frame, features)
            all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)  # (samples, frame, features)
    # all_data = all_data[:, 50:, :]
    return all_data, features


def arr_to_pd(data: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    data_dict = {
        feature_names[i]: (['sample', 'frame'], data[:, :, i])
        for i in range(len(feature_names))
    }
    ds = xr.Dataset(data_vars=data_dict,
                    coords={
                        'sample': np.arange(data.shape[0]),
                        'frame': np.arange(data.shape[1]),
                    })
    return ds


if __name__ == '__main__':
    h5_paths = ["C:\\Users\\php23rjb\\Downloads\\A.h5"]
    data, names = load_h5(h5_paths)
    ds = arr_to_pd(data, names)
