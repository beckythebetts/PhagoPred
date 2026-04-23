"""Functions to prodice exploratory graphs of multivariate time series"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, ccf, arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import MissingDataError
from tqdm import tqdm
import h5py
import umap
import pandas as pd
from sklearn.decomposition import PCA

from .data_loading import load_h5, arr_to_pd


@dataclass
class FeatureFits:
    """Store linear trend data and ARMA fit paramters for each time series"""
    m: float | None = None
    c: float | None = None
    p: int | list[int] | None = None
    q: int | list[int] | None = None
    adf_pvals: list[float] | None = None
    mean: float | None = None
    std: float | None = None

    def __str__(self):
        return asdict(self)


def differnce_ds(ds: xr.Dataset) -> None:
    ds_diff = ds.diff('frame')
    ds_diff = ds_diff.dropna('frame')
    return ds_diff


def smooth_ds(ds: xr.Dataset, window_size: int = 10) -> None:
    # WARNING: introduces auto correlation
    for feature_name, da in tqdm(ds.items(), desc='Smoothing'):
        ds[feature_name] = da.rolling(frame=window_size, center=True).mean()


def standardise_ds(ds: xr.Dataset, fits: FeatureFits) -> None:
    for feature_name, da in tqdm(ds.items(), desc='Standardising'):
        mean = da.mean()
        std = da.std()

        ds[feature_name] = (da - mean) / std
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFits
            fits[feature_name].mean = mean
            fits[feature_name].std = std


def subtract_linear_trend(ds: xr.Dataset,
                          fits: dict[str, FeatureFits] | None = None):
    """Fit a linear trend to each feature of the dataset and subtract."""
    for feature_name, da in tqdm(ds.items(), desc='Linear trend fitting'):
        frames = da.coords['frame']
        all_frames = []
        all_vals = []
        for _, sample in da.groupby('sample'):
            all_frames.extend(list(frames))
            all_vals.extend(list(sample.values[0]))
        # print(len(all_frames), len(all_vals))
        A = np.vstack([all_frames, np.ones(len(all_frames))]).T
        m, c = np.linalg.lstsq(A, all_vals)[0]
        ds[feature_name] = da - (m * da.coords['frame'].values + c)
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFits()
            setattr(fits[feature_name], 'm', m)
            setattr(fits[feature_name], 'c', c)


def fit_arma(ds: xr.Dataset,
             fits: dict[str, FeatureFits] | None = None) -> None:
    for feature_name, da in tqdm(ds.items(), desc='ARMA trend fitting'):
        frames = da.coords['frame']
        all_frames = []
        all_vals = []
        for _, sample in da.groupby('sample'):
            assert not np.isnan(sample.values[0]).any()
            results = arma_order_select_ic(
                sample.values[0],
                max_ar=4,
                max_ma=2,
                ic=['aic', 'bic'],
                #    fit_kw={'method': 'css'},
            )
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFits()
            setattr(fits[feature_name], 'm', m)
            setattr(fits[feature_name], 'c', c)
            print(results)
            # print(results.bic_min_order)
            # print(results.aic_min_order)


def subtract_arima(ds: xr.Dataset,
                   fits: dict[str, FeatureFits] | None = None) -> None:
    for feature_name, da in tqdm(ds.items(), desc='ARIMA fitting'):
        new_da = da.copy()
        for sample_id, sample_da in da.groupby('sample'):
            values = sample_da.values[0]
            nan_mask = ~np.isnan(values)
            clean_values = values[nan_mask]

            if len(clean_values) < 10:
                continue

            # Select ARMA order via AIC
            try:
                order_res = arma_order_select_ic(clean_values,
                                                 max_ar=4,
                                                 max_ma=2,
                                                 ic='aic')
                p, q = order_res.aic_min_order
            except Exception:
                p, q = 1, 0

            # Standardize for numerical stability, then fit ARIMA
            mu, sigma = clean_values.mean(), clean_values.std()
            scaled = (clean_values -
                      mu) / sigma if sigma > 0 else clean_values - mu
            try:
                model_fit = ARIMA(endog=scaled,
                                  order=(p, 0, q)).fit(maxiter=500)
                resid = model_fit.resid
            except Exception:
                resid = scaled - scaled.mean()

            # Restore NaN positions
            result = np.full(len(values), np.nan)
            result[nan_mask] = resid
            new_da.loc[{'sample': sample_id}] = result

            if fits is not None:
                if feature_name not in fits:
                    fits[feature_name] = FeatureFits()
                fits[feature_name].p = p
                fits[feature_name].q = q

        ds[feature_name] = new_da


def stationarity_test(ds: xr.Dataset,
                      fits: dict[str, FeatureFits] | None = None):
    for feature_name, da in ds.items():
        fig, axs = plt.subplots(5, 2, figsize=(10, 10))
        adf_stats = []
        pval_stats = []

        all_values = []
        all_rolling_averages = []
        all_rolling_stds = []
        for _, sample_da in da.groupby('sample'):
            # print(sample_da.values, len(sample_da.values))
            values = sample_da.values[0]
            frames = sample_da.coords['frame']
            nan_mask = ~np.isnan(values)
            values = values[nan_mask]
            frames = frames[nan_mask]
            all_values.append(values)
            # values = np.diff(values, prepend=[0])
            rolling_average = sample_da.rolling(
                frame=20, center=True).mean()[0][nan_mask]
            all_rolling_averages.append(rolling_average)
            rolling_std = sample_da.rolling(frame=20,
                                            center=True).std()[0][nan_mask]
            all_rolling_stds.append(rolling_std)
            axs[0, 0].plot(frames, values, color='k', alpha=0.5)

            axs[1, 0].plot(frames, rolling_average, color='k', alpha=0.5)

            axs[2, 0].plot(frames, rolling_std, color='k', alpha=0.5)

            plot_acf(values,
                     ax=axs[3, 0],
                     lags=40,
                     use_vlines=False,
                     color='k',
                     alpha=None,
                     marker=',',
                     title='Autocorrelation')
            plot_pacf(values,
                      ax=axs[3, 1],
                      lags=40,
                      use_vlines=False,
                      color='k',
                      alpha=None,
                      marker=',',
                      title='Partial Autocorrelation')
            try:
                adf, pval = adfuller(values)[0:2]
                adf_stats.append(adf)
                pval_stats.append(pval)
            except (ValueError,
                    # MissingDataError,
                    ):
                adf, pval = 0, 0

        adf_stats = get_percentiles(np.array(adf_stats), 5, 95)
        pval_precnetiles = get_percentiles(np.array(pval_stats), 5, 95)

        plot_percentiles(axs[0, 1], *get_percentiles(all_values, 5, 95, 0),
                         'k')
        plot_percentiles(axs[1, 1],
                         *get_percentiles(all_rolling_averages, 5, 95, 0), 'k')
        plot_percentiles(axs[2, 1], *get_percentiles(all_rolling_stds, 5, 95,
                                                     0), 'k')
        axs[0, 0].set_title('Raw Values')
        axs[1, 0].set_title('Rolling average (window = 20 frames)')
        axs[2, 0].set_title('Rolling STD (window = 20 frames)')
        axs[3, 0].set_title('Autocorrelation')
        axs[3, 1].set_title('Partial Autocorrelation')
        axs[4, 1].set_title('ADF test P-values')
        axs[3, 0].axhline(0, color='k', linewidth=0.5)
        axs[3, 0].axhline(1.96 * (len(values))**(-1 / 2),
                          color='k',
                          linewidth=0.5,
                          linestyle='--')
        axs[3, 0].axhline(-1.96 * (len(values))**(-1 / 2),
                          color='k',
                          linewidth=0.5,
                          linestyle='--')
        axs[3, 1].axhline(0, color='k', linewidth=0.5)
        axs[3, 1].axhline(1.96 * (len(values))**(-1 / 2),
                          color='k',
                          linewidth=0.5,
                          linestyle='--')
        axs[3, 1].axhline(-1.96 * (len(values))**(-1 / 2),
                          color='k',
                          linewidth=0.5,
                          linestyle='--')
        axs[4, 1].hist(pval_stats, bins=20, color='k')
        fig.suptitle(f'{feature_name} | '
                     f'ADF: {format_percentiles(*adf_stats)} | '
                     f'p: {format_percentiles(*pval_precnetiles)}')
        if fits is not None:
            if feature_name not in fits.keys():
                fits[feature_name] = FeatureFits
            setattr(fits[feature_name], 'adf_pvals', pval_stats)
        plt.tight_layout()
        plt.show()


def get_percentiles(array: np.ndarray,
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


def plot_cross_correlations(ds: xr.Dataset,
                            lags: int = 1,
                            vmin: float = -1,
                            vmax: float = 1,
                            cmap: str = 'coolwarm') -> None:
    # === NEED TO FIT AND SUBTRACT ARMA MODEL FIRST, TO DEAL WITH AUTOCORRELATIONS ===
    #
    num_samples = ds.dims['sample']
    ccfs = np.zeros((len(ds), len(ds), lags, num_samples))
    for s, (_, sample_ds) in enumerate(ds.groupby('sample')):
        for i, da_1 in enumerate(sample_ds.values()):
            for j, da_2 in enumerate(sample_ds.values()):
                ccf_ij = ccf(da_1.values[0], da_2.values[0], nlags=lags)
                ccfs[i, j, :, s] = ccf_ij
    mean_ccfs = np.nanmean(ccfs, axis=-1)
    fig, axs = plt.subplots(1, lags, figsize=(10, 10))
    for i in range(lags):
        im = axs[i].matshow(mean_ccfs[:, :, i],
                            vmin=vmin,
                            vmax=vmax,
                            cmap=cmap)
    fig.colorbar(im, ax=axs[-1])
    # axs[i].set_xticklabels(list(ds.keys()), rotation=45)
    # axs[i].set_yticklabels(list(ds.keys()))
    plt.tight_layout()
    plt.show()


def fit_ma1_model(feature_da: xr.DataArray) -> tuple[float, np.ndarray]:
    """
    Fit an MA(1) model using rho(1) = theta / (1 + theta**2).
    Returns the pooled theta estimate and per-sample ACF(1) values.
    """
    acf_1_vals = []
    for _, sample_da in feature_da.groupby('sample'):
        acf_1 = acf(sample_da.values.flatten(), nlags=1)[1]
        # if not np.isnan(acf_1):
        acf_1_vals.append(acf_1)

    acf_1_vals = np.array(acf_1_vals)
    rho_1 = np.mean(acf_1_vals)

    # Solve theta^2 - (1/rho_1)*theta + 1 = 0, pick |theta| < 1
    discriminant = 1 - 4 * rho_1**2
    if discriminant < 0:
        theta = np.sign(rho_1)  # fallback if no real solution
    else:
        theta_1 = (1 + np.sqrt(discriminant)) / (2 * rho_1)
        theta_2 = (1 - np.sqrt(discriminant)) / (2 * rho_1)
        theta = theta_1 if abs(theta_1) < 1 else theta_2

    return theta, acf_1_vals


def plot_ma1_fits(ds: xr.Dataset) -> pd.DataFrame:
    """
    For each feature, fit MA(1) and plot histogram of per-sample ACF(1) values
    with the pooled estimate and implied theta marked.
    """
    features = list(ds.data_vars)
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(6, 3 * n_features))
    if n_features == 1:
        axes = [axes]

    all_acf1 = {}
    for ax, feature_name in zip(axes, features):
        theta, acf_1_vals = fit_ma1_model(ds[feature_name])
        all_acf1[feature_name] = acf_1_vals

        ax.hist(acf_1_vals, bins=20, alpha=0.7)
        ax.axvline(np.mean(acf_1_vals),
                   color='r',
                   linestyle='--',
                   label=f'mean ACF(1) = {np.mean(acf_1_vals):.2f}')
        ax.axvline(np.median(acf_1_vals),
                   color='orange',
                   linestyle='--',
                   label=f'median ACF(1) = {np.median(acf_1_vals):.2f}')
        ax.set_title(f'{feature_name} | fitted θ = {theta:.2f}')
        ax.set_xlabel('ACF(1)')
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Return dataframe for downstream use e.g. UMAP
    return pd.DataFrame(all_acf1)


def plot_umap(acf1_df: pd.Dataframe) -> None:
    acf1_df = acf1_df.dropna()
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(acf1_df)

    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()


def plot_pca(acf1_df: pd.Dataframe) -> None:
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(acf1_df.dropna())
    print(pca.explained_variance_ratio_)
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, s=10)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.show()

    loading_df = pd.DataFrame(
        pca.components_.T,  # shape (n_features, n_components)
        index=acf1_df.columns,
        columns=['PC1', 'PC2'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pc in zip(axes, ['PC1', 'PC2']):
        loading_df[pc].sort_values().plot.barh(ax=ax)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(
            f'{pc} loadings ({pca.explained_variance_ratio_[int(pc[-1])-1]:.1%} variance)'
        )
        ax.set_xlabel('Loading')

    plt.tight_layout()
    plt.show()


def get_ma1_residuals(feature_da: xr.DataArray) -> xr.DataArray:
    residuals_list = []

    for sample_id, sample_da in feature_da.groupby('sample'):

        sample_da = sample_da.squeeze('sample')
        vals = sample_da.values

        if np.var(vals) == 0 or np.isnan(vals).any() or len(vals) < 10:
            continue

        rho_1 = acf(vals, nlags=1)[1]

        if np.isnan(rho_1):
            continue

        discriminant = 1 - 4 * rho_1**2
        if discriminant < 0:
            theta = np.sign(rho_1)
            print(theta)
        else:
            theta_1 = (1 + np.sqrt(discriminant)) / (2 * rho_1)
            theta_2 = (1 - np.sqrt(discriminant)) / (2 * rho_1)
            print(theta_1, theta_2)
            theta = theta_1 if abs(theta_1) < 1 else theta_2

        residuals = np.zeros_like(vals)
        for t in range(1, len(vals)):
            residuals[t] = vals[t] - theta * residuals[t - 1]

        # Now coords and dims match the squeezed 1D sample_da
        residuals_da = xr.DataArray(residuals,
                                    coords=sample_da.coords,
                                    dims=sample_da.dims)
        residuals_list.append(residuals_da)

    return xr.concat(residuals_list, dim='sample')


# def get_arima_residuals(feature_da: xr.DataArray) -> xr.DataArray:


def get_all_residuals(ds: xr.Dataset,
                      get_residuals_func: callable) -> xr.Dataset:
    return xr.Dataset({
        feature_name: get_residuals_func(da)
        for feature_name, da in tqdm(ds.items(), desc='Fitting MA(1)')
    })


if __name__ == '__main__':
    # h5_paths = [
    #     "C:\\Users\\php23rjb\\Downloads\\A.h5",
    #     "C:\\Users\\php23rjb\\Downloads\\E.h5",
    #     "C:\\Users\\php23rjb\\Downloads\\C.h5",
    #     "C:\\Users\\php23rjb\\Downloads\\D.h5"
    # ]
    # h5_paths = Path('PhagoPred') /
    fits = {}
    # for path in h5_paths:
    #     with h5py.File(path) as f:
    #         ds = f['Cells']['Phase']['Alive Phagocytes within 100 pixels'][:]
    #         above_threshold = ds > 100
    #         print(path, np.where(above_threshold))
    # ar, names =
    ds = arr_to_pd(*load_h5(h5_paths))
    # subtract_linear_trend(ds, fits)
    # smooth_ds(ds) # Creates high autocorrelation -> fails ADF test
    # differnce_ds(ds)
    standardise_ds(ds, fits)

    # ds = get_all_residuals(ds)
    # df = plot_ma1_fits(ds)
    # plot_pca(df)
    # for feature_name, da in ds.items():
    #     print(feature_name, fit_ma_1_model(da))
    # fit_ma_1_model(ds)
    # subtract_arma(ds, fits)
    # print(fits)

    # subtract_arima(ds, fits)
    stationarity_test(ds, fits)
    # plot_cross_correlations(ds, lags=5)
