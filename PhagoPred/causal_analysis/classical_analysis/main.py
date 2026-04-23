from pathlib import Path
from typing import Literal
import warnings

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import h5py

from ..data_loading import load_h5, arr_to_xrds, write_metadata
from .plots import plot_stationarity_test
from .utils import standardise_da, differnce_xr


def test_differences(
        ds: xr.Dataset,
        d: int,
        save_dir: Path,
        adf_pval_thresh: float = 0.05) -> dict[str, dict[str, float]]:
    results_dict = {}
    for feature_name, da in tqdm(
            ds.items(), desc='Checking stationarity over differences'):
        temp_da = da.copy()
        fig, axs = plt.subplots(6, d, figsize=(5 * d, 2.5 * 6))
        best_d = None
        for d_val in range(d):
            adf_vals, _, _ = plot_stationarity_test(temp_da,
                                                    list(axs[:, d_val]),
                                                    title=f'd = {d_val}')
            if np.mean(adf_vals) < adf_pval_thresh and best_d is None:
                results_dict[feature_name] = {
                    'd': d_val,
                    'mean': mean,
                    'std': std
                }
                best_d = d_val
            temp_da = differnce_xr(temp_da)
            temp_da, mean, std = standardise_da(temp_da)
        fig.suptitle(feature_name)
        plt.tight_layout()
        plt.savefig(save_dir / f'{feature_name}_differences.png')
        plt.close()
    return results_dict


def _estimate_arma_order(
    da: xr.Dataset,
    d: int,
    mean: float,
    std: float,
    ic: Literal['aic', 'bic'] = 'bic',
    max_ar: int = 2,
    max_ma: int = 2,
) -> dict[str, list[tuple[int, int]]]:
    ps = []
    qs = []
    for _ in range(d):
        da = differnce_xr(da)
    da = standardise_da(da, mean, std)[0]
    for _, sample_da in tqdm(da.groupby('sample')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.simplefilter('ignore', UserWarning)
            results = arma_order_select_ic(sample_da.values.squeeze(),
                                           max_ar=max_ar,
                                           max_ma=max_ma,
                                           ic=ic)
            p, q = getattr(results, f'{ic}_min_order')
            ps.append(p)
            qs.append(q)
    return ps, qs


def estimate_arma_order(
    h5_path=Path,
    ic: Literal['aic', 'bic'] = 'bic',
    max_ar: int = 2,
    max_ma: int = 1,
) -> None:
    with h5py.File(h5_path, 'r+') as f:
        features_group = f['Cells/Phase']
        for feature_name in f['Cells/Phase'].keys():
            if 'd' not in features_group[feature_name].attrs:
                continue
            print(f'Estimating ARIMA order for {feature_name}')
            d = features_group[feature_name].attrs['d']
            mean = features_group[feature_name].attrs['mean']
            std = features_group[feature_name].attrs['std']

            data = features_group[feature_name][:]

            num_frames, num_samples = data.shape
            da = xr.DataArray(
                data=data,
                coords={
                    'frame': np.arange(num_frames),
                    'sample': np.arange(num_samples)
                },
            )

            if feature_name == 'Speed':
                da.loc[{'frame': 0}] = 1.0
            nan_mask = da.isnull().any(dim='frame')
            da_clean = da.sel(sample=~nan_mask)
            ps, qs = _estimate_arma_order(da_clean, d, mean, std, ic, max_ar,
                                          max_ma)

            # arima_group = f.require_group(f'ARIMA/fits/{feature_name}')
            # arima_group.require_dataset('p_order', data=ps)
            # arima_group.create_dataset('q_order', data=qs)

            p_dataset = f.require_dataset(f'ARIMA/fits/{feature_name}/p_order',
                                          shape=(num_samples, ),
                                          dtype=np.float32,
                                          maxshape=(None, ))
            q_dataset = f.require_dataset(f'ARIMA/fits/{feature_name}/q_order',
                                          shape=(num_samples, ),
                                          dtype=np.float32,
                                          maxshape=(None, ))
            all_ps = np.full(num_samples, fill_value=np.nan)
            all_qs = np.full(num_samples, fill_value=np.nan)

            all_ps[~nan_mask] = ps
            all_qs[~nan_mask] = qs

            p_dataset[:] = all_ps
            q_dataset[:] = all_qs


def main():
    h5_paths = list((Path('PhagoPred') / 'Datasets' / '06_03').glob('*.h5'))

    # # 1. Load data to xarray
    # ds = arr_to_xrds(*load_h5(h5_paths))

    # # 2. Find minimum d value for staionarity (per feature)
    # results_dict = test_differences(ds,
    #                                 d=3,
    #                                 save_dir=Path('temp') /
    #                                 'classical_analysis')
    # for feat_name, feat_dict in results_dict.items():
    #     for key, val in feat_dict.items():
    #         write_metadata(h5_paths, f'Cells/Phase/{feat_name}', key, val)

    # 3. Fit ARIMA model to each cell, and get residuals
    for h5_path in h5_paths:
        print(f'Processing file {h5_path.name}')
        with h5py.File(h5_path, 'r+') as f:
            if 'ARIMA' in f.keys():
                del f['ARIMA']
        estimate_arma_order(h5_path)


if __name__ == '__main__':
    main()
