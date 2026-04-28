from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import xarray as xr
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.tsa.stattools import adfuller, acf, pacf, ccf

from .utils import get_percentiles

plt.rcParams["font.family"] = 'serif'


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


def _compute_ccfs(ds: xr.Dataset, lags: int) -> tuple[np.ndarray, list[str]]:
    feature_names = list(ds.data_vars)
    n_features = len(feature_names)
    n_lags = lags + 1  # lags 0..lags inclusive
    num_samples = ds.dims['sample']
    all_ccfs = np.full((n_features, n_features, n_lags, num_samples), np.nan)
    for s, (_, sample_ds) in enumerate(ds.groupby('sample')):
        arrays = [da.values.squeeze() for da in sample_ds.values()]
        for i in range(n_features):
            for j in range(n_features):
                all_ccfs[i, j, :, s] = ccf(arrays[i],
                                           arrays[j],
                                           nlags=lags + 1)[:n_lags]
    return all_ccfs, feature_names


def plot_ccf_distributions(ds: xr.Dataset,
                           lags: int = 10,
                           save_path: Path = None) -> None:
    all_ccfs, feature_names = _compute_ccfs(ds, lags)
    n_features = len(feature_names)
    n_lags = lags + 1
    lag_positions = list(range(n_lags))

    fig, axs = plt.subplots(n_features,
                            n_features,
                            figsize=(2.2 * n_features, 2 * n_features),
                            sharex=True,
                            sharey=True)
    for i in range(n_features):
        for j in range(n_features):
            ax = axs[i, j]
            data = [
                all_ccfs[i, j, l][~np.isnan(all_ccfs[i, j, l])]
                for l in range(n_lags)
            ]
            parts = ax.violinplot(data,
                                  positions=lag_positions,
                                  widths=0.7,
                                  showmedians=True,
                                  showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor('steelblue')
                pc.set_alpha(0.5)
            parts['cmedians'].set_color('k')
            ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
            ax.set_title(f'{i} → {j}', fontsize=7, pad=2)
            ax.tick_params(labelsize=5)
            if i == n_features - 1:
                ax.set_xticks(lag_positions)
                ax.set_xticklabels(lag_positions, fontsize=5)
            if j == 0:
                ax.set_ylabel('CCF', fontsize=6)

    legend_text = '\n'.join(f'{k}: {name}'
                            for k, name in enumerate(feature_names))
    fig.text(1.01,
             0.5,
             legend_text,
             va='center',
             ha='left',
             fontsize=7,
             transform=fig.transFigure,
             family='monospace')
    fig.suptitle(
        'CCF distributions across cells  (row i leads col j by lag frames)',
        fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_cross_correlations(ds: xr.Dataset,
                            lags: int = 10,
                            vmin: float = -1,
                            vmax: float = 1,
                            alpha: float = 0.05,
                            cmap: str = 'coolwarm',
                            save_path: Path = None) -> None:
    feature_names = list(ds.data_vars)
    n_features = len(feature_names)

    all_ccfs, _ = _compute_ccfs(ds, lags)
    n_lags = lags + 1

    mean_ccfs = np.nanmean(all_ccfs, axis=-1)

    # Wilcoxon signed-rank test per (feature_i, feature_j, lag)
    pvals = np.full((n_features, n_features, n_lags), np.nan)
    for i in range(n_features):
        for j in range(n_features):
            for l in range(n_lags):
                vals = all_ccfs[i, j, l, :]
                vals = vals[~np.isnan(vals)]
                if len(vals) >= 10:
                    try:
                        pvals[i, j, l] = wilcoxon(vals).pvalue
                    except ValueError:
                        pass
    sig_mask = pvals < alpha

    lag_labels = list(range(0, lags + 1))
    # tick_labels = [str(i) for i in range(n_features)]
    legend_text = '\n'.join(f'{i}: {name}'
                            for i, name in enumerate(feature_names))

    # Layout: heatmap columns + narrow legend column, thin colorbar row at bottom
    width_ratios = [1] * n_lags + [0.4]
    height_ratios = [10, 0.4]

    def _make_fig():
        fig = plt.figure(figsize=(2.5 * n_lags + 2, 4))
        gs = GridSpec(2,
                      n_lags + 1,
                      figure=fig,
                      width_ratios=width_ratios,
                      height_ratios=height_ratios,
                      hspace=0.15,
                      wspace=0.15)
        hm_axs = [fig.add_subplot(gs[0, l]) for l in range(n_lags)]
        leg_ax = fig.add_subplot(gs[:, -1])
        cbar_ax = fig.add_subplot(gs[1, :n_lags])
        leg_ax.axis('off')
        leg_ax.text(0.05,
                    0.98,
                    legend_text,
                    va='top',
                    ha='left',
                    fontsize=7,
                    transform=leg_ax.transAxes,
                    family='serif')
        return fig, hm_axs, cbar_ax

    def _format_ax(ax, lag_val, first):
        ax.set_title(f'Lag {lag_val}', fontsize=8)
        # ax.set_xticks(range(n_features))
        # ax.set_xticklabels(tick_labels, fontsize=7)
        # ax.set_yticks(range(n_features))
        # ax.set_yticklabels(tick_labels if first else [], fontsize=7)

    # --- Figure 1: mean CCF, non-significant cells greyed out ---
    fig1, axs1, cbar1_ax = _make_fig()
    for l, lag_val in enumerate(lag_labels):
        display = np.where(sig_mask[:, :, l], mean_ccfs[:, :, l], np.nan)
        im1 = axs1[l].matshow(display, vmin=vmin, vmax=vmax, cmap=cmap)
        # axs1[l].matshow(np.where(~sig_mask[:, :, l], 0.5, np.nan),
        #                 vmin=0,
        #                 vmax=1,
        #                 cmap='Greys',
        #                 alpha=0.4)
        _format_ax(axs1[l], lag_val, first=(l == 0))
    fig1.colorbar(im1,
                  cax=cbar1_ax,
                  orientation='horizontal',
                  label='Cross-correlation')
    fig1.suptitle(f'Mean CCF of ARIMA residuals (grey = p≥{alpha}, Wilcoxon)',
                  fontsize=9)

    # --- Figure 2: -log10(p-value) heatmap ---
    log_pvals = -np.log10(np.clip(pvals, 1e-300, 1.0))
    sig_threshold = -np.log10(alpha)
    fig2, axs2, cbar2_ax = _make_fig()
    for l, lag_val in enumerate(lag_labels):
        im2 = axs2[l].matshow(log_pvals[:, :, l],
                              vmin=0,
                              vmax=sig_threshold * 3,
                              cmap='YlOrRd')
        _format_ax(axs2[l], lag_val, first=(l == 0))
    fig2.colorbar(im2,
                  cax=cbar2_ax,
                  orientation='horizontal',
                  label=f'−log₁₀(p)   [dashed = {alpha} threshold]')
    cbar2_ax.axvline(sig_threshold, color='k', linestyle='--', linewidth=0.8)
    fig2.suptitle('Wilcoxon p-values for CCF across samples', fontsize=9)

    if save_path is None:
        plt.show()
    else:
        fig1.savefig(save_path, bbox_inches='tight')
        pval_path = save_path.with_stem(save_path.stem + '_pvalues')
        fig2.savefig(pval_path, bbox_inches='tight')
    plt.close('all')
