from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import os
import json
from typing import Tuple
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf, ccf, arma_order_select_ic
from tqdm import tqdm

from .data_loading import load_h5, arr_to_pd

plt.rcParams["font.family"] = 'serif'


@dataclass
class FeatureFit:
    mean: float | None = None
    std: float | None = None
    d: int | None = None
    p: int | None = None
    q: int | None = None

    def asdict(self):
        return asdict(self)


def differnce_ds(ds: xr.Dataset | xr.DataArray) -> None:
    ds_diff = ds.diff('frame')
    ds_diff = ds_diff.dropna('frame')
    return ds_diff


def standardise(data: xr.Dataset | xr.DataArray,
                fits: FeatureFit | None = None) -> xr.DataArray | xr.Dataset:
    if isinstance(data, xr.DataArray):
        return _standardise_da(data)[0]
    standardised_ds = data.copy()
    for feature_name, da in tqdm(data.items(), desc='Standardising'):
        da, mean, std = _standardise_da(da)
        standardised_ds[feature_name] = da
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFit()
            fits[feature_name].mean = mean
            fits[feature_name].std = std
    return standardised_ds


def _standardise_da(da: xr.DataArray) -> tuple[xr.DataArray, float, float]:
    mean = da.mean()
    std = da.std()
    return (da - mean) / std, float(mean), float(std)


def plot_stationarity_test(
        da: xr.DataArray,
        axs: list[plt.Axes],
        title: str,
        rolling_window_size: int = 20) -> tuple[list, list, list]:
    ax_raw, ax_mean, ax_std, ax_acf, ax_pcf, ax_adf = axs

    mean_da = da.rolling(frame=rolling_window_size, center=True).mean()
    std_da = da.rolling(frame=rolling_window_size, center=True).std()

    plot_percentiles(
        ax_raw,
        *get_percentiles(da.values, 5, 95, axis=da.dims.index('sample')),
        colour='k',
    )
    plot_percentiles(
        ax_mean,
        *get_percentiles(mean_da.values, 5, 95, axis=da.dims.index('sample')),
        colour='k',
    )
    plot_percentiles(
        ax_std,
        *get_percentiles(std_da.values, 5, 95, axis=da.dims.index('sample')),
        colour='k',
    )

    # acf_plot_params = {
    #     'lags': 40,
    #     'use_vlines': False,
    #     'color': 'k',
    #     'alpha': None,
    #     'marker': ',',
    #     'title': '',
    # }
    all_acf_vals = []
    all_pacf_vals = []
    adf_pvals = []

    for _, sample_da in da.groupby('sample'):
        acf_vals = acf(sample_da.values[0], nlags=20)
        # print(acf_vals)
        ax_acf.scatter(np.arange(len(acf_vals)),
                       acf_vals,
                       marker='.',
                       color='k')

        pacf_vals = pacf(sample_da.values[0], nlags=20)
        ax_pcf.scatter(np.arange(len(pacf_vals)),
                       pacf_vals,
                       marker='.',
                       color='k')
        all_acf_vals.append(acf_vals)
        all_pacf_vals.append(pacf_vals)
        # plot_acf(sample_da.values[0], ax_acf, **acf_plot_params)
        # plot_pacf(sample_da.values[0], ax_pcf, **acf_plot_params)
        try:
            adf_pval = adfuller(sample_da.values[0])[1]
        except ValueError:
            adf_pval = 1.0
        adf_pvals.append(adf_pval)
    ax_adf.hist(adf_pvals, bins=50, color='k')

    ax_raw.set_title(title)
    ax_raw.set_xlabel('Frame')
    ax_raw.set_ylabel('Raw Values')

    ax_mean.set_xlabel('Frame')
    ax_mean.set_ylabel(f'Rolling average\nwindow = {rolling_window_size}')

    ax_std.set_xlabel('Frame')
    ax_std.set_ylabel(
        f'Rolling Standard Deviation\nwindow = {rolling_window_size}')

    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('Autocorrelation')

    ax_pcf.set_xlabel('Lag')
    ax_pcf.set_ylabel('Parital Autocorrelation')

    ax_adf.set_xlabel('Augmented Dickey-Fuller unit root test\nP values')
    ax_adf.set_ylabel('Frequency')

    add_acf_confidence_bounds(ax_acf, da.sizes['sample'])
    add_acf_confidence_bounds(ax_pcf, da.sizes['sample'])

    return adf_pvals, all_acf_vals, all_pacf_vals


def add_acf_confidence_bounds(ax: plt.Axes, num_vals: int) -> None:
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axhline(1.96 * (num_vals)**(-1 / 2),
               color='k',
               linewidth=0.5,
               linestyle='--')
    ax.axhline(-1.96 * (num_vals)**(-1 / 2),
               color='k',
               linewidth=0.5,
               linestyle='--')


def get_percentiles(array: np.ndarray | xr.DataArray,
                    lower_percentile: int,
                    upper_percentile: int,
                    axis: int = None) -> tuple[float, float, float]:
    lower = np.percentile(array, lower_percentile, axis=axis)
    median = np.median(array, axis=axis)
    upper = np.percentile(array, upper_percentile, axis=axis)

    return lower, median, upper


def plot_percentiles(ax: plt.Axes, lower: np.ndarray, median: np.ndarray,
                     upper: np.ndarray, colour: str) -> None:
    x = np.arange(len(median))
    ax.fill_between(x, lower, upper, color=colour, alpha=0.3, edgecolor=None)
    ax.plot(x, median, color=colour)


def format_percentiles(lower: float, median: float, upper: float) -> str:
    return f'{median:.3f} [{lower:.3f}, {upper:.3f}]'


def _test_differnces_da(da: xr.DataArray,
                        d: int,
                        save_path: Path,
                        feature_name: str,
                        adf_pval_thresh: float = 0.05) -> Tuple[int, int, int]:

    temp_da = da.copy()
    fig, axs = plt.subplots(6, d, figsize=(5 * d, 2.5 * 6))
    min_d = None
    p = None
    q = None
    for d_val in range(d):
        adf_vals, acf_vals, pacf_vals = plot_stationarity_test(
            temp_da, list(axs[:, d_val]), title=f'd = {d_val}')
        if np.mean(adf_vals) < adf_pval_thresh and min_d is None:
            min_d = d_val
            # p, q = arma_order_select_ic()
            # p, q = estimate_p_q(acf_vals, pacf_vals, nobs=da.sizes['frame'])
            p, q = estimate_p_q(
                da, d_val, save_path.parent / f'p_q_hist_{feature_name}.png')
        temp_da = differnce_ds(temp_da)
        temp_da = standardise(temp_da)
    fig.suptitle(feature_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return min_d, p, q


def estimate_p_q(da: xr.DataArray,
                 d: int,
                 save_path: Path | None = None,
                 max_ar: int = 2,
                 max_ma: int = 1) -> tuple[int, int]:
    aic_ps, aic_qs = [], []
    bic_ps, bic_qs = [], []
    aic_scores, bic_scores = [], []

    da = da.copy()
    for _ in range(d):
        da = differnce_ds(da)

    for _, sample_da in tqdm(da.groupby('sample'),
                             desc='Estimating ARMA order'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.simplefilter('ignore', UserWarning)
            results = arma_order_select_ic(sample_da.values[0],
                                           max_ar=max_ar,
                                           max_ma=max_ma,
                                           ic=['aic', 'bic'])
        aic_p, aic_q = results.aic_min_order
        bic_p, bic_q = results.bic_min_order
        aic_ps.append(aic_p)
        aic_qs.append(aic_q)
        bic_ps.append(bic_p)
        bic_qs.append(bic_q)
        aic_scores.append(results.aic.values)
        bic_scores.append(results.bic.values)

    aic_scores = np.array(aic_scores)
    bic_scores = np.array(bic_scores)

    if save_path is not None:
        fig, axs = plt.subplots(3, 2, figsize=(10, 12))
        for col, (scores, ps, qs, ic_name) in enumerate([
            (aic_scores, aic_ps, aic_qs, 'AIC'),
            (bic_scores, bic_ps, bic_qs, 'BIC'),
        ]):
            plt_ic_matrix(scores,
                          axs[0, col],
                          title=f'{ic_name} matrix (d={d})')
            axs[1, col].hist(ps, bins=np.arange(max_ar + 2) - 0.5, color='k')
            axs[1, col].set_xlabel('p (AR order)')
            axs[1, col].set_ylabel('Count')
            axs[1, col].set_title(f'{ic_name} selected p')
            axs[1, col].set_xticks(range(max_ar + 1))
            axs[2, col].hist(qs, bins=np.arange(max_ma + 2) - 0.5, color='k')
            axs[2, col].set_xlabel('q (MA order)')
            axs[2, col].set_ylabel('Count')
            axs[2, col].set_title(f'{ic_name} selected q')
            axs[2, col].set_xticks(range(max_ma + 1))
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # Select best order from pooled AIC matrix
    best = np.unravel_index(np.nanargmin(np.nanmean(aic_scores, axis=0)),
                            aic_scores.shape[1:])
    return int(best[0]), int(best[1])


def plt_ic_matrix(scores: np.ndarray, ax: plt.Axes, title: str = '') -> None:

    mean = np.nanmean(scores, axis=0)
    std = np.nanstd(scores, axis=0)

    # Normalise so minimum = 0 (ΔAIC), easier to read
    mean -= np.nanmin(mean)

    im = ax.imshow(mean, cmap='viridis')

    # Overlay std as text in each cell
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            ax.text(j,
                    i,
                    f'{mean[i,j]:.1f}\n±{std[i,j]:.1f}',
                    ha='center',
                    va='center',
                    fontsize=7)

    ax.set_xlabel('q (MA order)')
    ax.set_ylabel('p (AR order)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='ΔAIC')


# def estimate_p_q(acf_vals: list, pacf_vals: list,
#                  nobs: int) -> Tuple[int, int]:
#     ci = 1.96 / np.sqrt(nobs)
#     mean_acf_vals = np.mean(acf_vals, axis=0)
#     mean_pacf_vals = np.mean(pacf_vals, axis=0)
#     p = np.argma
#     p = next((i for i, v in enumerate(mean_pacf_vals) if abs(v) < ci), 0)
#     q = next((i for i, v in enumerate(mean_acf_vals) if abs(v) < ci), 0)

#     return p, q


def test_differences(ds: xr.Dataset,
                     d: int,
                     save_dir: Path,
                     fits: dict,
                     adf_pval_thresh: float = 0.05):
    for feature_name, da in tqdm(
            ds.items(), desc='Checking stationarity over differences'):
        min_d, p, q = _test_differnces_da(
            da,
            d=d,
            save_path=save_dir / f'differences_test_{feature_name}.png',
            feature_name=feature_name,
            adf_pval_thresh=adf_pval_thresh,
        )
        print('\n', min_d, p, q)
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFit()
            fits[feature_name].d = min_d
            fits[feature_name].p = p
            fits[feature_name].q = q


# def _fit_arima_da(da: xr.DataArray, d: int):

# def estimate_arma_order(da: xr.DataArray, fits: dict)
if __name__ == '__main__':
    h5_paths = [
        "C:\\Users\\php23rjb\\Downloads\\A.h5",
        "C:\\Users\\php23rjb\\Downloads\\E.h5",
        "C:\\Users\\php23rjb\\Downloads\\C.h5",
        "C:\\Users\\php23rjb\\Downloads\\D.h5"
    ]

    test_dir = Path('C:\\Users\\php23rjb\\Documents\\PhagoPred\\temp'
                    ) / 'classical_analysis'
    os.makedirs(test_dir, exist_ok=True)
    fits = {}

    ds = arr_to_pd(*load_h5(h5_paths))
    standardise(ds, fits)
    test_differences(ds, d=3, save_dir=test_dir, fits=fits)

    fits = {k: v.asdict() for k, v in fits.items()}
    with (test_dir / 'fits.json').open('w') as f:
        json.dump(fits, f)
