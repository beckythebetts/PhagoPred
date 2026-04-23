from __future__ import annotations

import numpy as np
import xarray as xr


def get_percentiles(array: np.ndarray | xr.DataArray,
                    lower_percentile: int,
                    upper_percentile: int,
                    axis: int = None) -> tuple[float, float, float]:
    lower = np.percentile(array, lower_percentile, axis=axis)
    median = np.median(array, axis=axis)
    upper = np.percentile(array, upper_percentile, axis=axis)

    return lower, median, upper


def format_percentiles(lower: float, median: float, upper: float) -> str:
    return f'{median:.3f} [{lower:.3f}, {upper:.3f}]'


def differnce_xr(ds: xr.Dataset | xr.DataArray) -> None:
    ds_diff = ds.diff('frame')
    ds_diff = ds_diff.dropna('frame')
    return ds_diff


# def standardise(
#     data: xr.Dataset | xr.DataArray,
#     return_vals: bool = False
#     # fits: FeatureFit | None = None,
# ) -> xr.DataArray | xr.Dataset:
#     if isinstance(data, xr.DataArray):
#         return standardise_da(data)[0]
#     standardised_ds = data.copy()
#     for feature_name, da in tqdm(data.items(), desc='Standardising'):
#         da, mean, std = standardise_da(da)
#         standardised_ds[feature_name] = da
#         if return_vals:
#             return standardised_ds, mean, std
#     return standardised_ds


def standardise_da(
        da: xr.DataArray,
        mean: float | None = None,
        std: float | None = None) -> tuple[xr.DataArray, float, float]:
    if mean is None:
        mean = da.mean()
    if std is None:
        std = da.std()
    return (da - mean) / std, float(mean), float(std)
