import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf, pacf

from .utils import get_percentiles


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


def plot_percentiles(ax: plt.Axes, lower: np.ndarray, median: np.ndarray,
                     upper: np.ndarray, colour: str) -> None:
    x = np.arange(len(median))
    ax.fill_between(x, lower, upper, color=colour, alpha=0.3, edgecolor=None)
    ax.plot(x, median, color=colour)
