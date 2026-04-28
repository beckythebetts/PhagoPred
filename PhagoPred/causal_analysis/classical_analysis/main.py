from pathlib import Path
from typing import Literal
import warnings

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
import h5py

from PhagoPred.utils.logger import get_logger
from ..data_loading import load_h5, arr_to_xrds, write_metadata
from .plots import plot_stationarity_test, plot_cross_correlations, plot_ccf_distributions
from .utils import standardise_da, differnce_xr

log = get_logger()


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
    h5_path: Path,
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
            nan_mask = da.isnull().any(dim='frame').values
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


def _fit_arima_model(da: xr.Dataset, p: int, q: int) -> tuple:
    """Fit an ARIMA model to a single sample.
    Returns (ar_coeffs, ma_coeffs, residuals, ar_bse, ma_bse, aic, bic, llf, sigma2, mae).
    """
    num_frames = da.sizes['frame']
    nan_scalars = (np.nan, np.nan, np.nan, np.nan, np.nan)

    if np.isnan(p) or np.isnan(q):
        return (np.array([np.nan]), np.array([np.nan]),
                np.full(num_frames, np.nan), np.array([np.nan]),
                np.array([np.nan]), *nan_scalars)
    p, q = int(p), int(q)
    if p == 0 and q == 0:
        residuals = da.values.squeeze()
        mae = float(np.nanmean(np.abs(residuals)))
        return (np.array([]), np.array([]), residuals, np.array([]),
                np.array([]), *(*nan_scalars[:4], mae))
    data = da.values.squeeze()
    if np.sum(~np.isnan(data)) < p + q + 2:
        return (np.full(p,
                        np.nan), np.full(q,
                                         np.nan), np.full(num_frames, np.nan),
                np.full(p, np.nan), np.full(q, np.nan), *nan_scalars)
    try:
        arima = ARIMA(data, order=(p, 0, q))
        arima_results: ARIMAResults = arima.fit()
        ar_coeffs = arima_results.arparams
        ma_coeffs = arima_results.maparams
        arima_residuals = arima_results.resid
        ar_bse = arima_results.bse[:p]
        ma_bse = arima_results.bse[p:p + q]
        aic = float(arima_results.aic)
        bic = float(arima_results.bic)
        llf = float(arima_results.llf)
        sigma2 = float(arima_results.params[-1])
        mae = float(np.mean(np.abs(arima_residuals)))
    except (np.linalg.LinAlgError, Exception):
        return (np.full(p,
                        np.nan), np.full(q,
                                         np.nan), np.full(num_frames, np.nan),
                np.full(p, np.nan), np.full(q, np.nan), *nan_scalars)

    return ar_coeffs, ma_coeffs, arima_residuals, ar_bse, ma_bse, aic, bic, llf, sigma2, mae


def fit_arima_model(h5_path: Path):
    with h5py.File(h5_path, 'r+') as f:
        features_group = f['Cells/Phase']
        arima_group = f['ARIMA/fits']

        for feature_name in f['Cells/Phase'].keys():
            if 'd' not in features_group[feature_name].attrs:
                continue
            d = features_group[feature_name].attrs['d']
            mean = features_group[feature_name].attrs['mean']
            std = features_group[feature_name].attrs['std']

            data = features_group[feature_name][:]
            num_frames, num_samples = data.shape

            max_p = np.nanmax(arima_group[feature_name]['p_order'][:])
            max_q = np.nanmax(arima_group[feature_name]['q_order'][:])

            # Create datasets for MA, AR coeffs — delete and recreate if shape mismatch
            def require_dataset_shape(group, key, shape, dtype):
                full_key = f'{feature_name}/{key}'
                if full_key in group:
                    if group[full_key].shape != shape:
                        del group[full_key]
                return group.require_dataset(full_key,
                                             shape=shape,
                                             dtype=dtype)

            ar_coeffs_ds = require_dataset_shape(arima_group, 'ar_coeffs',
                                                 (num_samples, max_p),
                                                 np.float32)
            ma_coeffs_ds = require_dataset_shape(arima_group, 'ma_coeffs',
                                                 (num_samples, max_q),
                                                 np.float32)
            ar_bse_ds = require_dataset_shape(arima_group, 'ar_bse',
                                              (num_samples, max_p), np.float32)
            ma_bse_ds = require_dataset_shape(arima_group, 'ma_bse',
                                              (num_samples, max_q), np.float32)
            for scalar_key in ('aic', 'bic', 'llf', 'sigma2', 'mae'):
                require_dataset_shape(arima_group, scalar_key, (num_samples, ),
                                      np.float32)

            da = xr.DataArray(
                data=data,
                coords={
                    'frame': np.arange(num_frames),
                    'sample': np.arange(num_samples)
                },
            )

            nan_mask = np.isnan(arima_group[feature_name]['p_order'][:])
            clean_indices = np.where(~nan_mask)[0]
            da_clean = da.sel(sample=clean_indices)
            for _ in range(d):
                da_clean = differnce_xr(da_clean)
            da_clean = standardise_da(da_clean, mean, std)[0]

            num_residual_frames = da_clean.sizes['frame']

            arima_residuals_ds = require_dataset_shape(
                arima_group, 'residuals', (num_residual_frames, num_samples),
                np.float32)

            ar_coeffs = []
            ma_coeffs = []
            residuals = []
            ar_bse_list = []
            ma_bse_list = []
            aic_list = []
            bic_list = []
            llf_list = []
            sigma2_list = []
            mae_list = []
            for sample_idx in tqdm(range(num_samples)):
                if nan_mask[sample_idx]:
                    ar_coeffs.append(np.full(int(max_p), np.nan))
                    ma_coeffs.append(np.full(int(max_q), np.nan))
                    residuals.append(np.full(num_residual_frames, np.nan))
                    ar_bse_list.append(np.full(int(max_p), np.nan))
                    ma_bse_list.append(np.full(int(max_q), np.nan))
                    aic_list.append(np.nan)
                    bic_list.append(np.nan)
                    llf_list.append(np.nan)
                    sigma2_list.append(np.nan)
                    mae_list.append(np.nan)
                    continue

                sample_da = da_clean.sel(sample=sample_idx)
                p_order = arima_group[feature_name]['p_order'][sample_idx]
                q_order = arima_group[feature_name]['q_order'][sample_idx]

                (sample_ar_coeffs, sample_ma_coeffs, sample_residuals,
                 sample_ar_bse, sample_ma_bse, aic, bic, llf, sigma2,
                 mae) = _fit_arima_model(sample_da, p_order, q_order)
                sample_ar_coeffs = np.concatenate(
                    (sample_ar_coeffs,
                     np.zeros(int(max_p) - len(sample_ar_coeffs))))
                sample_ma_coeffs = np.concatenate(
                    (sample_ma_coeffs,
                     np.zeros(int(max_q) - len(sample_ma_coeffs))))
                sample_ar_bse = np.concatenate(
                    (sample_ar_bse, np.zeros(int(max_p) - len(sample_ar_bse))))
                sample_ma_bse = np.concatenate(
                    (sample_ma_bse, np.zeros(int(max_q) - len(sample_ma_bse))))

                ar_coeffs.append(sample_ar_coeffs)
                ma_coeffs.append(sample_ma_coeffs)
                residuals.append(sample_residuals)
                ar_bse_list.append(sample_ar_bse)
                ma_bse_list.append(sample_ma_bse)
                aic_list.append(aic)
                bic_list.append(bic)
                llf_list.append(llf)
                sigma2_list.append(sigma2)
                mae_list.append(mae)

            ar_coeffs = np.stack(ar_coeffs, axis=0)
            ma_coeffs = np.stack(ma_coeffs, axis=0)
            residuals = np.stack(residuals, axis=1)

            ar_coeffs_ds[:] = ar_coeffs
            ma_coeffs_ds[:] = ma_coeffs
            arima_residuals_ds[:] = residuals
            ar_bse_ds[:] = np.stack(ar_bse_list, axis=0)
            ma_bse_ds[:] = np.stack(ma_bse_list, axis=0)
            arima_group[f'{feature_name}/aic'][:] = aic_list
            arima_group[f'{feature_name}/bic'][:] = bic_list
            arima_group[f'{feature_name}/llf'][:] = llf_list
            arima_group[f'{feature_name}/sigma2'][:] = sigma2_list
            arima_group[f'{feature_name}/mae'][:] = mae_list


def main():
    h5_paths = list((Path('PhagoPred') / 'Datasets' / '06_03').glob('*.h5'))

    # 1. Load data to xarray
    ds = arr_to_xrds(*load_h5(h5_paths))

    # # 2. Find minimum d value for staionarity (per feature)
    # log.info('Finding minimum d for stationarity')
    # results_dict = test_differences(ds,
    #                                 d=3,
    #                                 save_dir=Path('temp') /
    #                                 'classical_analysis')
    # for feat_name, feat_dict in results_dict.items():
    #     for key, val in feat_dict.items():
    #         write_metadata(h5_paths, f'Cells/Phase/{feat_name}', key, val)

    # 3. Fit ARIMA model to each cell, and get residuals
    # log.info('Finding ARIMA params + residuals')
    # for h5_path in h5_paths:
    #     print(f'Processing file {h5_path.name}')
    #     # with h5py.File(h5_path, 'r+') as f:
    #     #     if 'ARIMA' in f.keys():
    #     #         del f['ARIMA']
    #     log.info(f'Finding ARIMA order for file at {h5_path}')
    #     # estimate_arma_order(h5_path)
    #     log.info(f'Fitting ARIMA for file at {h5_path}')
    #     fit_arima_model(h5_path)

    # 4. Plot cross correlations of residuals
    residuals_ds = arr_to_xrds(*load_h5(h5_paths, arima_residuals=True))
    save_dir = Path('temp') / 'classical_analysis'
    plot_cross_correlations(residuals_ds,
                            lags=5,
                            save_path=save_dir / 'cross_correlations.png')
    # plot_ccf_distributions(residuals_ds,
    #                        lags=5,
    #                        save_path=save_dir / 'ccf_distributions.png')


if __name__ == '__main__':
    main()
