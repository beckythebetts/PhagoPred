from typing import Optional, Union
import sys
import textwrap
from itertools import combinations

from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from scipy.stats import (wasserstein_distance, ks_2samp, mannwhitneyu,
                         bootstrap)
from matplotlib.colors import LinearSegmentedColormap

from PhagoPred import SETTINGS
from PhagoPred.feature_extraction.extract_features import CellType

plt.rcParams["font.family"] = 'serif'


def plot_cell_features(
        h5_file: Path,
        cell_idx: int,
        first_frame: int,
        last_frame: int,
        save_as: Path,
        feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Plot time series of features for given cell index. If list of features is not given, plot all features."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(h5_file, 'r') as f:
        if feature_names is None:
            feature_names = list(f['Cells']['Phase'].keys())

        n = len(feature_names)
        fig, axs = plt.subplots(n, sharex=True, figsize=(10, max(0.6 * n, 4)))
        if n == 1:
            axs = [axs]

        for i, feature_name in enumerate(feature_names):
            feature_values = f['Cells']['Phase'][feature_name][
                first_frame:last_frame, cell_idx]
            axs[i].plot(range(first_frame, last_frame),
                        feature_values,
                        color='k',
                        linewidth=1)
            wrapped = textwrap.fill(feature_name, width=16)
            axs[i].set_ylabel(wrapped,
                              rotation=0,
                              ha='right',
                              va='center',
                              labelpad=6,
                              fontsize=8)
            axs[i].tick_params(axis='both', labelsize=7)
            axs[i].grid(alpha=0.4)
            axs[i].set_xlim(left=first_frame, right=last_frame - 1)

        fig.suptitle(f'Cell {cell_idx}', fontsize=11, fontweight='bold')
        axs[-1].set_xlabel('Frame', fontsize=9)

    plt.subplots_adjust(left=0.22, hspace=0.08, top=0.97)
    plt.savefig(save_as, bbox_inches='tight', dpi=150)

    return fig


def plot_average_cell_features(
        save_as: Path,
        first_frame: Optional[int] = None,
        last_frame: Optional[int] = None,
        feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Plot time series of features averaged over all cells. If list of features is not given, plot all features."""
    plt.rcParams["font.family"] = 'serif'

    with h5py.File(SETTINGS.DATASET, 'r') as f:
        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()
        if first_frame is None:
            first_frame = 0
        if last_frame is None:
            last_frame = f['Cells']['Phase'][feature_names[0]].shape[0]

        fig, axs = plt.subplots(len(feature_names),
                                sharex=True,
                                figsize=(10, 10))
        for i, feature_name in enumerate(tqdm(feature_names)):
            feature_values = f['Cells']['Phase'][feature_name][
                first_frame:last_frame]
            # feature_means = np.nanmean(feature_values, axis=1)
            # feature_stds = np.nanstd(feature_values, axis=1)

            # Compute nanmean and nanstd safely
            feature_means = np.full(feature_values.shape[0], np.nan)
            feature_stds = np.full(feature_values.shape[0], np.nan)

            for t in range(feature_values.shape[0]):
                if np.isnan(feature_values[t]).all():
                    continue  # Leave mean and std as NaN for that timepoint
                feature_means[t] = np.nanmean(feature_values[t])
                feature_stds[t] = np.nanstd(feature_values[t])

            axs[i].plot(range(first_frame, last_frame), feature_means)
            axs[i].fill_between(range(first_frame, last_frame),
                                feature_means - feature_stds,
                                feature_means + feature_stds,
                                alpha=0.5,
                                edgecolor=None)

            axs[i].set_ylabel(feature_name, rotation=45, labelpad=20)
            axs[i].grid()
            axs[i].set_xlim(left=first_frame, right=last_frame - 1)

        axs[-1].set(xlabel='Frame')

    plt.savefig(save_as)

    return fig


def plot_percentile_cell_features_multi(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    first_frame: Optional[int] = None,
    last_frame: Optional[int] = None,
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot time series of features showing 5th, 50th (median), and 95th percentiles over multiple datasets."""

    plt.rcParams["font.family"] = 'serif'
    num_files = len(hdf5_files)

    # Basic validation
    if labels is not None and len(labels) != num_files:
        raise ValueError("Length of labels must match number of HDF5 files")

    # Colors for plotting
    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(num_files)]

    # Determine feature names from the first file if not given
    if feature_names is None:
        with h5py.File(hdf5_files[0], 'r') as f:
            feature_names = list(f['Cells']['Phase'].keys())

    # Determine frames to plot from the first file if not given
    if first_frame is None or last_frame is None:
        with h5py.File(hdf5_files[0], 'r') as f:
            total_frames = f['Cells']['Phase']['Area'].shape[0]
            if first_frame is None:
                first_frame = 0
            if last_frame is None:
                last_frame = total_frames

    fig, axs = plt.subplots(len(feature_names),
                            sharex=True,
                            figsize=(12, 3 * len(feature_names)))

    for i, feature_name in enumerate(tqdm(feature_names)):
        ax = axs[i] if len(feature_names) > 1 else axs

        for file_idx, file_path in enumerate(hdf5_files):
            with h5py.File(file_path, 'r') as f:
                feature_values = f['Cells']['Phase'][feature_name][
                    first_frame:last_frame]

                # Initialize arrays for percentiles
                p5 = np.full(feature_values.shape[0], np.nan)
                p50 = np.full(feature_values.shape[0], np.nan)
                p95 = np.full(feature_values.shape[0], np.nan)

                p5 = np.nanpercentile(feature_values, 25, axis=1)
                p50 = np.nanpercentile(feature_values, 50, axis=1)
                p95 = np.nanpercentile(feature_values, 75, axis=1)
                # for t in range(feature_values.shape[0]):
                #     data_t = feature_values[t]
                #     if np.isnan(data_t).all():
                #         continue
                #     p5[t] = np.nanpercentile(data_t, 5)
                #     p50[t] = np.nanpercentile(data_t, 50)
                #     p95[t] = np.nanpercentile(data_t, 95)
            print(feature_name, p50.shape)
            # Plot median line
            ax.plot(
                range(first_frame, last_frame),
                p50,
                color=colors[file_idx],
                label=labels[file_idx] if labels else f'Dataset {file_idx+1}')
            # Fill between 5th and 95th percentiles
            ax.fill_between(range(first_frame, last_frame),
                            p5,
                            p95,
                            color=colors[file_idx],
                            edgecolor='none',
                            alpha=0.3)

        ax.set_ylabel(feature_name, rotation=45, labelpad=20)
        ax.grid(True)
        ax.set_xlim(left=first_frame, right=last_frame - 1)

    axs[-1].set_xlabel('Frame')

    # Add legend to the top plot
    axs[0].legend(loc='upper right', fontsize=12, frameon=False)

    plt.tight_layout()
    plt.savefig(save_as)
    return fig


def plot_feature_correlations(
        save_as: Path,
        feature_names: Optional[list] = None) -> 'matplotlib.figure.Figure':
    """Show scatter plots and correlations between each feature, and histograms of each feature."""
    with h5py.File(SETTINGS.DATASET, 'r') as f:

        if feature_names is None:
            feature_names = f['Cells']['Phase'].keys()

        num_features = len(feature_names)
        plt.rcParams["font.family"] = 'serif'
        fig, axs = plt.subplots(len(feature_names),
                                len(feature_names),
                                figsize=(num_features, num_features))
        cmap = plt.get_cmap('viridis_r')

        for i, feature_name_i in enumerate(tqdm(feature_names)):
            for j, feature_name_j in enumerate(feature_names):

                # sys.stdout.write(f'\rPLotting Correlations: Progress {(i*num_features+j+1)/(num_features**2) * 100:.0f}%')
                # sys.stdout.flush()

                feature_i = remove_outliers(
                    f['Cells']['Phase'][feature_name_i][:].flatten())
                feature_j = remove_outliers(
                    f['Cells']['Phase'][feature_name_j][:].flatten())

                R = corr_coefficient(feature_i, feature_j)
                color = cmap((R + 1) / 2)

                if i == 0:
                    axs[i, j].set_title(feature_name_j, rotation=45)

                if j == 0:
                    axs[i, j].set_ylabel(feature_name_i,
                                         rotation=45,
                                         labelpad=30)

                if j != 0 and i != j:
                    axs[i, j].set_yticklabels([])

                if i != len(feature_names) - 1:
                    axs[i, j].set_xticklabels([])

                # plot

                if i > j:
                    axs[i, j].scatter(feature_j,
                                      feature_i,
                                      s=0.1,
                                      linewidths=0,
                                      color=color)
                    # axs[i, j].grid()

                if i == j:
                    axs[i, j].hist(feature_i, bins=100, color='k')
                    # axs[i, j].grid()

                if i < j:
                    axs[i, j].text(0.5,
                                   0.5,
                                   f'R = \n{R:.2f}',
                                   ha='center',
                                   va='center',
                                   transform=axs[i, j].transAxes,
                                   fontsize=14,
                                   fontweight='bold',
                                   color=color)
                    axs[i, j].axis('off')
    plt.savefig(save_as)

    return fig


def plot_random_subset(ax,
                       x,
                       y,
                       color,
                       label,
                       subset_size=1000,
                       marker='o',
                       alpha=0.6,
                       s=10):
    # Ensure subset_size does not exceed data size
    n = len(x)
    if subset_size < n:
        indices = np.random.choice(n, subset_size, replace=False)
        x_sub = x[indices]
        y_sub = y[indices]
    else:
        x_sub = x
        y_sub = y

    ax.scatter(x_sub,
               y_sub,
               s=s,
               color=color,
               alpha=alpha,
               label=label,
               marker=marker)


def plot_qq(ax,
            reference: np.ndarray,
            sample: np.ndarray,
            color,
            n_quantiles: int = 500,
            label: Optional[str] = None) -> None:
    """Quantile-quantile plot of ``sample`` against ``reference``.

    Identical distributions lie on y = x, a shift lifts the curve off it and a
    change in spread tilts it. Unlike overlaid histograms this does not hide
    tail differences, which is what makes it readable when two distributions
    are nearly the same.
    """
    quantiles = np.linspace(0.001, 0.999, n_quantiles)
    reference_q = np.nanquantile(reference, quantiles)
    sample_q = np.nanquantile(sample, quantiles)

    ax.plot(reference_q, sample_q, color=color, linewidth=1.5, label=label)


def plot_density_difference(ax,
                            x_a: np.ndarray,
                            y_a: np.ndarray,
                            x_b: np.ndarray,
                            y_b: np.ndarray,
                            cmap,
                            x_range: tuple,
                            y_range: tuple,
                            bins: int = 60,
                            robust_percentile: float = 99) -> None:
    """Image the difference between two 2D densities on a shared grid.

    Both groups are binned identically and normalised to densities, so the
    colour shows where in feature space one group has proportionally more cells
    than the other. Regions where the two agree come out blank, which is the
    point: overlaid scatter spends all its ink on the agreement.

    The colour limit is a high percentile of the difference rather than its
    max, because otherwise the single densest bin sets the scale and washes out
    every other difference in the panel.
    """
    mask_a = ~np.isnan(x_a) & ~np.isnan(y_a)
    mask_b = ~np.isnan(x_b) & ~np.isnan(y_b)
    x_a, y_a = x_a[mask_a], y_a[mask_a]
    x_b, y_b = x_b[mask_b], y_b[mask_b]

    if len(x_a) == 0 or len(x_b) == 0:
        return

    grid_range = [list(x_range), list(y_range)]
    hist_a, x_edges, y_edges = np.histogram2d(x_a,
                                              y_a,
                                              bins=bins,
                                              range=grid_range,
                                              density=True)
    hist_b, _, _ = np.histogram2d(x_b,
                                  y_b,
                                  bins=bins,
                                  range=grid_range,
                                  density=True)

    difference = hist_a - hist_b
    occupied = (hist_a + hist_b) > 0
    limit = (np.percentile(np.abs(difference[occupied]), robust_percentile)
             if occupied.any() else 0.0)
    if not limit > 0:
        limit = 1.0

    ax.imshow(difference.T,
              origin='lower',
              cmap=cmap,
              vmin=-limit,
              vmax=limit,
              extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
              aspect='auto',
              interpolation='nearest')


def plot_interleaved_scatter(ax,
                             xs: list[np.ndarray],
                             ys: list[np.ndarray],
                             colors: list,
                             max_points: int = 100000,
                             alpha: float = 0.1,
                             s: float = 1) -> None:
    """Scatter several groups with the draw order shuffled between them.

    Plotting one group after another puts the last group on top everywhere,
    which makes it look denser than it is. Interleaving the points removes that
    artefact.
    """
    subsets_x, subsets_y, point_colors = [], [], []
    for x, y, color in zip(xs, ys, colors):
        if len(x) > max_points:
            idx = np.random.choice(len(x), max_points, replace=False)
            x, y = x[idx], y[idx]
        subsets_x.append(x)
        subsets_y.append(y)
        point_colors.append(np.tile(np.asarray(color), (len(x), 1)))

    x_all = np.concatenate(subsets_x)
    y_all = np.concatenate(subsets_y)
    colors_all = np.concatenate(point_colors)

    order = np.random.permutation(len(x_all))
    ax.scatter(x_all[order],
               y_all[order],
               s=s,
               c=colors_all[order],
               alpha=alpha,
               marker='.')


def plot_feature_correlations_multi(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Plot scatter plots and histograms for features across multiple datasets.

    Files sharing the same entry in ``labels`` are pooled into one group, drawn
    in a single colour with a single legend entry. Repeating a label is how
    several hdf5 files are grouped together, e.g.
    ``labels=['No Fluor'] * 3 + ['Fluor'] * 6``.
    """

    # Validate inputs
    num_files = len(hdf5_files)
    if labels is None:
        labels = [Path(file_path).stem for file_path in hdf5_files]
    if len(labels) != num_files:
        raise ValueError("Length of labels must match number of HDF5 files")

    # Group files by label, keeping the order the labels first appear in
    group_labels = list(dict.fromkeys(labels))
    grouped_files = [[
        file_path for file_path, label in zip(hdf5_files, labels)
        if label == group_label
    ] for group_label in group_labels]
    num_groups = len(group_labels)

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(num_groups)]

    # Diverging map running between the two group colours, so a red region of a
    # difference panel means "denser in the red group" without a second key
    if num_groups == 2:
        difference_cmap = LinearSegmentedColormap.from_list(
            'group_difference', [colors[1], 'white', colors[0]])

    # Load all features, pooling the files belonging to each group
    all_data = []
    for group_files in grouped_files:
        group_dict = {}
        for file_path in group_files:
            with h5py.File(file_path, 'r') as f:
                if feature_names is None:
                    feature_names = list(f['Cells']['Phase'].keys())

                for feat in feature_names:
                    raw_data = f['Cells']['Phase'][feat][:]
                    flat = raw_data.flatten()
                    cleaned = remove_outliers(flat)
                    group_dict.setdefault(feat, []).append(cleaned)
        all_data.append({
            feat: np.concatenate(arrs)
            for feat, arrs in group_dict.items()
        })

    # One display range per feature, shared by every panel in its row/column so
    # that hiding the inner tick labels is honest
    feature_ranges = {}
    for feat in feature_names:
        pooled = np.concatenate([data[feat] for data in all_data])
        pooled = pooled[~np.isnan(pooled)]
        feature_ranges[feat] = tuple(np.percentile(pooled, (0.5, 99.5)))

    num_features = len(feature_names)
    fig, axs = plt.subplots(num_features,
                            num_features,
                            figsize=(num_features * 3, num_features * 3))
    plt.rcParams["font.family"] = 'serif'

    for i, feature_i in enumerate(tqdm(feature_names)):
        for j, feature_j in enumerate(feature_names):

            ax = axs[i, j]
            if i == 0:
                ax.set_title(feature_j, rotation=45, fontsize=20)
            if j == 0:
                ax.set_ylabel(feature_i, rotation=45, labelpad=30, fontsize=20)

            # Hide x/y ticks as appropriate
            if i != num_features - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            if i > j:
                if num_groups == 2:
                    # Lower triangle: where is one group denser than the other?
                    plot_density_difference(ax,
                                            all_data[0][feature_j],
                                            all_data[0][feature_i],
                                            all_data[1][feature_j],
                                            all_data[1][feature_i],
                                            cmap=difference_cmap,
                                            x_range=feature_ranges[feature_j],
                                            y_range=feature_ranges[feature_i])
                else:
                    plot_interleaved_scatter(
                        ax,
                        [all_data[k][feature_j] for k in range(num_groups)],
                        [all_data[k][feature_i]
                         for k in range(num_groups)], colors)
                ax.set_xlim(*feature_ranges[feature_j])
                ax.set_ylim(*feature_ranges[feature_i])
            elif i == j:
                if num_groups == 1:
                    ax.hist(all_data[0][feature_i],
                            bins=100,
                            color=colors[0],
                            alpha=0.5,
                            density=True)
                else:
                    # Diagonal: Q-Q of every group against the first one
                    reference = all_data[0][feature_i]
                    for k in range(1, num_groups):
                        plot_qq(ax, reference, all_data[k][feature_i],
                                colors[k])
                    lo, hi = feature_ranges[feature_i]
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)
                    ax.axline((lo, lo),
                              slope=1,
                              color='grey',
                              linestyle='--',
                              linewidth=0.8)
                    ax.text(0.04,
                            0.94,
                            'Q-Q',
                            transform=ax.transAxes,
                            ha='left',
                            va='top',
                            fontsize=10,
                            color='grey')
            else:
                # Upper triangle: how much the correlation itself differs
                if num_groups == 2:
                    r_a = corr_coefficient(all_data[0][feature_j],
                                           all_data[0][feature_i])
                    r_b = corr_coefficient(all_data[1][feature_j],
                                           all_data[1][feature_i])
                    ax.text(0.5,
                            0.6,
                            f'$\\Delta$R = {r_b - r_a:+.3f}',
                            ha='center',
                            va='center',
                            transform=ax.transAxes,
                            fontsize=20,
                            fontweight='bold',
                            color='black')
                    ax.text(0.5,
                            0.35,
                            f'{r_a:.2f} / {r_b:.2f}',
                            ha='center',
                            va='center',
                            transform=ax.transAxes,
                            fontsize=12,
                            color='grey')
                else:
                    for k in range(num_groups):
                        r = corr_coefficient(all_data[k][feature_j],
                                             all_data[k][feature_i])
                        ax.text(0.5,
                                0.5 + (num_groups - 1 - 2 * k) * 0.15,
                                f'R = {r:.2f}',
                                ha='center',
                                va='center',
                                transform=ax.transAxes,
                                fontsize=20,
                                fontweight='bold',
                                color=colors[k])
                ax.axis('off')

    # Reserve a fixed strip at the top for the legend and the panel key, so
    # neither lands on the rotated column titles
    figure_height = num_features * 3
    plt.tight_layout(rect=[0, 0, 1, 1 - 1.7 / figure_height])

    # Collect legend handles from the last diagonal plot
    handles = []
    for k in range(num_groups):
        handles.append(
            plt.Line2D([], [],
                       marker='o',
                       color=colors[k],
                       linestyle='None',
                       markersize=5,
                       label=group_labels[k]))

    # Add a single legend outside the plot grid
    fig.legend(handles=handles,
               labels=group_labels,
               loc='upper right',
               bbox_to_anchor=(0.995, 1 - 0.15 / figure_height),
               ncol=num_groups,
               fontsize=20,
               frameon=False)

    if num_groups == 2:
        fig.text(0.995,
                 1 - 0.7 / figure_height,
                 f'Lower: density difference, colour = denser group\n'
                 f'Diagonal: Q-Q, y = {group_labels[1]}, '
                 f'x = {group_labels[0]}\n'
                 f'Upper: $\\Delta$R = R({group_labels[1]}) '
                 f'- R({group_labels[0]})',
                 ha='right',
                 va='top',
                 fontsize=13,
                 color='dimgrey',
                 linespacing=1.6)

    plt.savefig(save_as)
    return fig


def significance_stars(p: float) -> str:
    """Conventional star annotation for a p-value."""
    for threshold, stars in ((0.0001, '****'), (0.001, '***'), (0.01, '**'),
                             (0.05, '*')):
        if p < threshold:
            return stars
    return 'ns'


def annotate_significance(ax,
                          x1: float,
                          x2: float,
                          y: float,
                          p: float,
                          bar_height: float,
                          show_p_value: bool = True,
                          fontsize: int = 9,
                          color: str = 'k') -> None:
    """Draw the usual bracket-and-stars comparison bar between two positions."""
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y],
            linewidth=1.2,
            color=color)

    stars = significance_stars(p)
    label = f'{stars}  (p = {p:.2g})' if show_p_value else stars
    ax.text((x1 + x2) / 2,
            y + bar_height,
            label,
            ha='center',
            va='bottom',
            fontsize=fontsize,
            color=color)


def bootstrap_difference_ci(a: np.ndarray,
                            b: np.ndarray,
                            confidence: float = 0.95,
                            n_resamples: int = 9999,
                            seed: int = 0) -> tuple[float, float, float]:
    """Difference in means of ``b`` and ``a`` with a bootstrap CI.

    Reported instead of a bare p-value because a non-significant test does not
    show two groups are the same, whereas the interval says how large a
    difference the data still allow. Applied to per-well summaries, so the
    interval reflects well-to-well variability rather than cell counts.
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    observed = np.mean(b) - np.mean(a) if len(a) and len(b) else np.nan

    if len(a) < 2 or len(b) < 2:
        return observed, np.nan, np.nan

    def statistic(x, y, axis=-1):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    for method in ('BCa', 'percentile'):
        try:
            result = bootstrap((a, b),
                               statistic,
                               method=method,
                               confidence_level=confidence,
                               n_resamples=n_resamples,
                               random_state=seed)
            return (observed, result.confidence_interval.low,
                    result.confidence_interval.high)
        except Exception:
            continue

    return observed, np.nan, np.nan


def plot_per_well_summary(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    seed: int = 0,
    equivalence_percent: Optional[float] = None,
) -> plt.Figure:
    """One point per well (hdf5 file), grouped by label.

    Treats the well, not the cell, as the unit of replication. Pooling millions
    of cells makes vanishingly small group differences look overwhelming, but
    cells within a well are not independent samples of "with fluorescence" -
    the wells are. The spread of the points within a group is the noise any
    real group difference has to beat, and it is readable directly off the
    plot.

    Top row is each well's median for a feature, middle row its interquartile
    range (i.e. whether the spread differs, not just the centre). Both are
    robust, so no outlier removal is applied. A box summarises each group, but
    the wells stay plotted on top of it: with ~12 per group, the box's five
    numbers would hide the very spread the plot exists to show.

    The bottom panel gives the group difference with a bootstrap confidence
    interval. The significance bars above answer "is there evidence of a
    difference"; only the interval answers "how big could the difference still
    be", which is the question a non-significant result leaves open.
    ``equivalence_percent`` optionally shades a band of differences small
    enough not to matter - choosing that bound is a scientific judgement, so
    there is no default.
    """
    plt.rcParams["font.family"] = 'serif'

    num_files = len(hdf5_files)
    if labels is None:
        labels = [Path(file_path).stem for file_path in hdf5_files]
    if len(labels) != num_files:
        raise ValueError("Length of labels must match number of HDF5 files")

    group_labels = list(dict.fromkeys(labels))
    num_groups = len(group_labels)
    group_indices = [group_labels.index(label) for label in labels]

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(num_groups)]

    if feature_names is None:
        with h5py.File(hdf5_files[0], 'r') as f:
            feature_names = list(f['Cells']['Phase'].keys())

    # One median and one IQR per well per feature
    medians = {feat: [] for feat in feature_names}
    iqrs = {feat: [] for feat in feature_names}
    for file_path in tqdm(hdf5_files, desc='Summarising wells'):
        with h5py.File(file_path, 'r') as f:
            for feat in feature_names:
                values = f['Cells']['Phase'][feat][:].ravel()
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    medians[feat].append(np.nan)
                    iqrs[feat].append(np.nan)
                    continue
                q1, q2, q3 = np.percentile(values, [25, 50, 75])
                medians[feat].append(q2)
                iqrs[feat].append(q3 - q1)

    num_features = len(feature_names)
    statistics = [('Median', medians), ('IQR', iqrs)]

    # The confidence intervals only make sense for a single comparison
    include_intervals = num_groups == 2
    forest_inches = max(2.2, 0.5 * len(statistics) * num_features + 1.0)
    panel_inches = 3.2

    fig = plt.figure(figsize=(2.9 * num_features, 2 * panel_inches +
                              (forest_inches if include_intervals else 0) + 1))
    grid = fig.add_gridspec(
        3 if include_intervals else 2,
        num_features,
        height_ratios=([panel_inches, panel_inches, forest_inches]
                       if include_intervals else [panel_inches, panel_inches]),
        hspace=0.55)
    axs = np.array(
        [[fig.add_subplot(grid[row, col]) for col in range(num_features)]
         for row in range(2)])

    rng = np.random.default_rng(seed)
    group_indices = np.array(group_indices)

    for row, (statistic, values_by_feature) in enumerate(statistics):
        for col, feat in enumerate(feature_names):
            ax = axs[row, col]
            values = np.array(values_by_feature[feat])

            group_values = [
                values[group_indices == k][np.isfinite(
                    values[group_indices == k])] for k in range(num_groups)
            ]

            # Box for the group, wells kept on top of it: with ~12 per group
            # the five numbers of a box would hide the spread that matters
            if all(len(y) > 0 for y in group_values):
                boxes = ax.boxplot(group_values,
                                   positions=range(num_groups),
                                   widths=0.5,
                                   showfliers=False,
                                   patch_artist=True,
                                   zorder=2)
                for k, patch in enumerate(boxes['boxes']):
                    patch.set_facecolor(colors[k])
                    patch.set_alpha(0.25)
                    patch.set_edgecolor(colors[k])
                for part in ('whiskers', 'caps'):
                    for item in boxes[part]:
                        item.set_color('k')
                for median_line in boxes['medians']:
                    median_line.set_color('k')
                    median_line.set_linewidth(2)

            for k, y in enumerate(group_values):
                # Jittered strip of the individual wells
                x = k + rng.uniform(-0.12, 0.12, len(y))
                ax.scatter(
                    x,
                    y,
                    color=colors[k],
                    s=35,
                    alpha=0.8,
                    edgecolor='none',
                    linewidth=0.5,
                    zorder=3,
                )

            ax.set_xticks(range(num_groups))
            ax.set_xticklabels([
                f'{label}\n(n = {len(y)})'
                for label, y in zip(group_labels, group_values)
            ],
                               fontsize=10)
            ax.set_xlim(-0.6, num_groups - 0.4)
            ax.grid(axis='y', alpha=0.3)

            if col == 0:
                ax.set_ylabel(f'Per-well {statistic}', fontsize=12)
            if row == 0:
                ax.set_title(textwrap.fill(feat, width=16), fontsize=12)

            # Tests compare wells, which is the honest n (wells, not cells)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                continue
            span = finite.max() - finite.min() or 1.0
            level = finite.max() + 0.12 * span

            for k1, k2 in combinations(range(num_groups), 2):
                a, b = group_values[k1], group_values[k2]
                if len(a) == 0 or len(b) == 0:
                    continue
                # p = mannwhitneyu(a, b).pvalue
                # annotate_significance(ax,
                #                       k1,
                #                       k2,
                #                       level,
                #                       p,
                #                       bar_height=0.035 * span)
                level += 0.2 * span

            ax.set_ylim(finite.min() - 0.1 * span, level)

    if include_intervals:
        ax_forest = fig.add_subplot(grid[2, :])
        markers = ['o', 's']

        for row, (statistic, values_by_feature) in enumerate(statistics):
            centres, lows, highs, positions = [], [], [], []

            for index, feat in enumerate(feature_names):
                values = np.array(values_by_feature[feat])
                a = values[group_indices == 0]
                b = values[group_indices == 1]

                difference, low, high = bootstrap_difference_ci(a,
                                                                b,
                                                                seed=seed)
                # As a percentage, so features on different scales compare
                scale = np.nanmean(np.concatenate([a, b]))
                if not np.isfinite(scale) or scale == 0:
                    continue
                centres.append(100 * difference / scale)
                lows.append(100 * (difference - low) / scale)
                highs.append(100 * (high - difference) / scale)
                positions.append(index + (row - 0.5) * 0.3)

            ax_forest.errorbar(centres,
                               positions,
                               xerr=[lows, highs],
                               fmt=markers[row],
                               color='k' if row == 0 else 'grey',
                               markersize=6,
                               capsize=4,
                               linewidth=1.4,
                               label=f'Per-well {statistic}')

        if equivalence_percent is not None:
            ax_forest.axvspan(-equivalence_percent,
                              equivalence_percent,
                              color='grey',
                              alpha=0.15,
                              zorder=0,
                              label=f'±{equivalence_percent:g}% bound')
        ax_forest.axvline(0, color='k', linewidth=1)

        ax_forest.set_yticks(range(num_features))
        ax_forest.set_yticklabels(feature_names, fontsize=11)
        ax_forest.set_ylim(num_features - 0.5, -0.5)
        ax_forest.set_xlabel(
            f'{group_labels[1]} - {group_labels[0]} '
            f'(% of pooled value, 95% bootstrap CI)',
            fontsize=11)
        ax_forest.set_title(
            'Difference between groups: an interval spanning 0 means no '
            'evidence of a difference,\nnot evidence that the groups match - '
            'read its width for how large a difference is still possible',
            fontsize=10,
            color='dimgrey')
        ax_forest.grid(axis='x', alpha=0.3)
        ax_forest.legend(fontsize=10, frameon=False, loc='best')

    handles = [
        plt.Line2D([], [],
                   marker='o',
                   color=colors[k],
                   linestyle='None',
                   markersize=8,
                   label=group_labels[k]) for k in range(num_groups)
    ]
    fig.legend(handles=handles,
               loc='upper right',
               bbox_to_anchor=(0.995, 0.995),
               ncol=num_groups,
               fontsize=11,
               frameon=False)

    fig.suptitle('Per well feature comparison', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_as, dpi=150, bbox_inches='tight')

    return fig


def plot_num_alive_cells(files: list = [SETTINGS.DATASET],
                         save_as: Optional[Path] = None,
                         first_frame: Optional[int] = None,
                         last_frame: Optional[int] = None,
                         time_step: int = 5,
                         labels: list = []) -> 'matplotlib.figure.Figure':
    """PLot number of alive cells at each frame."""
    plt.rcParams["font.family"] = 'serif'
    cmap = plt.get_cmap('Set1')
    colours = [cmap(i) for i in range(len(files))]
    fig, ax = plt.subplots(figsize=(10, 5))

    for file, colour, label in zip(files, colours, labels):
        with h5py.File(file, 'r') as f:
            file_first_frame = 0 if first_frame is None else first_frame
            file_last_frame = f['Cells']['Phase']['Area'].shape[
                0] if last_frame is None else last_frame

            first_frames = f['Cells']['Phase']['First Frame'][0]
            last_frames = f['Cells']['Phase']['CellDeath'][0]
            last_frames = np.where(np.isnan(last_frames),
                                   f['Cells']['Phase']['Last Frame'][0],
                                   last_frames)

            print(first_frames, last_frames)

            frames = np.arange(file_first_frame, file_last_frame)
            alive_mask = (first_frames[np.newaxis, :]
                          <= frames[:,
                                    np.newaxis]) & (last_frames[np.newaxis, :]
                                                    > frames[:, np.newaxis])

            num_alive = alive_mask.sum(axis=1)
            times = np.arange(file_first_frame, file_last_frame) * time_step
            ax.plot(times, num_alive, label=label, color=colour)
            # ax.plot(range(first_frame, last_frame), total_cells, label='Total Cells', color='black', linestyle='--')
        ax.set_xlabel('Time / minutes')
        ax.set_ylabel('Number of Detected Alive Cells')
        ax.legend()
        ax.grid()
    if save_as is not None:
        plt.savefig(save_as)

    return fig


def plot_cell_tracks(
    save_as: Path,
    first_frame: Optional[int] = None,
    last_frame: Optional[int] = None,
    background_frame: Optional[int] = None,
    hdf5_file: Optional[Union[str, Path]] = None,
    linewidth: float = 0.8,
) -> plt.Figure:
    """Plot the centroid tracks of each cell overlaid on a phase image.

    Tracks fade from transparent (oldest) to opaque (most recent). Each cell
    is assigned a distinct hue that cycles across all cells.

    Args:
        save_as: Path to save the figure.
        first_frame: First frame to include in tracks (default: 0).
        last_frame: Last frame to include in tracks (default: all frames).
        background_frame: Frame index to use as background phase image (default: first_frame).
        hdf5_file: Path to the HDF5 dataset (default: SETTINGS.DATASET).
        linewidth: Width of the track lines.
    """
    from matplotlib.collections import LineCollection

    if hdf5_file is None:
        hdf5_file = SETTINGS.DATASET

    with h5py.File(hdf5_file, 'r') as f:
        total_frames = f['Cells']['Phase']['X'].shape[0]
        if first_frame is None:
            first_frame = 0
        if last_frame is None:
            last_frame = total_frames
        if background_frame is None:
            background_frame = first_frame

        x_all = f['Cells']['Phase']['X'][first_frame:last_frame]  # (T, N)
        y_all = f['Cells']['Phase']['Y'][first_frame:last_frame]  # (T, N)
        phase_img = f['Images']['Phase'][background_frame]  # (H, W)

    num_cells = x_all.shape[1]
    num_frames = x_all.shape[0]
    # HSV hue spread evenly so colours are maximally distinct for any cell count
    hue_cmap = plt.get_cmap('hsv')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(phase_img, cmap='gray', interpolation='nearest')

    # Alpha ramp: oldest frame → 0, most recent → 1
    alpha_ramp = np.linspace(0.0, 1.0, num_frames)

    for cell_idx in range(num_cells):
        # X is stored as row (vertical), Y as column (horizontal) — swap for imshow axes
        x = x_all[:, cell_idx]
        y = y_all[:, cell_idx]
        valid = ~(np.isnan(x) | np.isnan(y))
        if valid.sum() < 2:
            continue

        r, g, b, _ = hue_cmap(cell_idx / num_cells)

        # Walk through contiguous valid runs and build a LineCollection per run
        frames = np.where(valid)[0]
        breaks = np.where(np.diff(frames) > 1)[0] + 1
        segments_idx = np.split(frames, breaks)

        for seg in segments_idx:
            if len(seg) < 2:
                continue
            # Build array of (N-1) line segments and matching RGBA colours
            pts = np.stack([y[seg], x[seg]], axis=1)  # (T, 2) — col, row
            segs = np.stack([pts[:-1], pts[1:]], axis=1)  # (T-1, 2, 2)
            # Alpha for each segment is the mean of its two endpoint alphas
            seg_alpha = (alpha_ramp[seg[:-1]] + alpha_ramp[seg[1:]]) / 2
            colors = np.column_stack([
                np.full(len(seg_alpha), r),
                np.full(len(seg_alpha), g),
                np.full(len(seg_alpha), b),
                seg_alpha,
            ])
            lc = LineCollection(segs, colors=colors, linewidths=linewidth)
            ax.add_collection(lc)

    # ax.set_title(f'Cell Tracks (frames {first_frame}–{last_frame - 1})',
    #              fontsize=12)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches='tight')
    # plt.imsave(save_as)
    return fig


def remove_outliers(arr):
    """
    Using interquartile range (+/- 1.5*q)
    """
    no_nan = arr[~np.isnan(arr)]
    q1 = np.percentile(no_nan, 0.25)
    q3 = np.percentile(no_nan, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    arr[(arr < lower_bound)] = np.nan
    arr[(arr > upper_bound)] = np.nan
    return arr


# def corr_coefficient(array1, array2):
#     """"
#     Deals with np.nan when calculating correlation coefficeinet (numpy)
#     """
#     mask = ~np.isnan(array1) & ~np.isnan(array2)
#     return np.corrcoef(array1[mask], array2[mask])[0, 1]


def corr_coefficient(array1, array2):
    """
    Calculate correlation coefficient handling NaNs and constant arrays.
    """
    mask = ~np.isnan(array1) & ~np.isnan(array2)
    filtered_x = array1[mask]
    filtered_y = array2[mask]

    if len(filtered_x) < 2:
        return np.nan  # Not enough data to compute correlation

    if np.std(filtered_x) == 0 or np.std(filtered_y) == 0:
        return np.nan  # Constant arrays => undefined correlation

    return np.corrcoef(filtered_x, filtered_y)[0, 1]


def plot_death_frames_hist(death_frames_txt: Path, save_as: Path):
    death_frames = pd.read_csv(death_frames_txt,
                               sep="|",
                               skiprows=2,
                               engine='python')
    death_frames = death_frames.iloc[:, 1:-1]
    death_frames.columns = ['Cell Idx', 'True', 'Predicted']
    death_frames = death_frames[[
        'Cell Idx', 'True', 'Predicted'
    ]].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    valid_rows = death_frames[
        death_frames['True'].str.isnumeric()
        & death_frames['Predicted'].str.isnumeric()].copy()

    valid_rows['True'] = valid_rows['True'].astype(int)
    valid_rows['Predicted'] = valid_rows['Predicted'].astype(int)

    errors = valid_rows['Predicted'] - valid_rows['True']
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Number of Cells')
    plt.title('Histogram of Cell Death Frame Prediction Errors')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_as)


def plot_two_death_frame_hists(death_frames_txt1: Path,
                               death_frames_txt2: Path, label1: str,
                               label2: str, save_as: Path):

    def load_and_compute_errors(path: Path):
        df = pd.read_csv(path, sep="|", skiprows=2, engine='python')
        df = df.iloc[:, 1:-1]
        df.columns = ['Cell Idx', 'True', 'Predicted']
        df = df[['Cell Idx', 'True', 'Predicted'
                 ]].applymap(lambda x: x.strip() if isinstance(x, str) else x)
        valid = df[df['True'].str.isnumeric()
                   & df['Predicted'].str.isnumeric()].copy()
        valid['True'] = valid['True'].astype(int)
        valid['Predicted'] = valid['Predicted'].astype(int)
        errors = valid['Predicted'] - valid['True']
        return errors

    errors1 = load_and_compute_errors(death_frames_txt1)
    errors2 = load_and_compute_errors(death_frames_txt2)
    all_errors = pd.concat([errors1, errors2])
    min_edge = all_errors.min()
    max_edge = all_errors.max()
    bins = 30  # or any number you choose
    bin_edges = np.linspace(min_edge, max_edge, bins + 1)

    plt.figure(figsize=(8, 6))
    plt.hist(errors1,
             bins=bin_edges,
             alpha=0.6,
             color='blue',
             edgecolor='black',
             label=label1)
    plt.hist(errors2,
             bins=bin_edges,
             alpha=0.6,
             color='red',
             edgecolor='black',
             label=label2)

    plt.xlabel('Prediction Error (Predicted - True)')
    plt.ylabel('Number of Cells')
    plt.title('Comparison of Cell Death Frame Prediction Errors')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def plot_death_frame(death_frames_txt: Path, save_as: Path):
    death_frames = pd.read_csv(death_frames_txt,
                               sep="|",
                               skiprows=2,
                               engine='python')
    death_frames = death_frames.iloc[:, 1:-1]
    death_frames.columns = ['Cell Idx', 'True', 'Predicted']
    death_frames = death_frames[[
        'Cell Idx', 'True', 'Predicted'
    ]].applymap(lambda x: x.strip() if isinstance(x, str) else x)

    valid_rows = death_frames[
        death_frames['True'].str.isnumeric()
        & death_frames['Predicted'].str.isnumeric()].copy()

    valid_rows['True'] = valid_rows['True'].astype(int)
    valid_rows['Predicted'] = valid_rows['Predicted'].astype(int)

    plt.figure(figsize=(8, 6))
    plt.scatter(valid_rows['True'],
                valid_rows['Predicted'],
                color='blue',
                label='Cell Death Prediction',
                marker='.')
    plt.plot([valid_rows['True'].min(), valid_rows['True'].max()],
             [valid_rows['True'].min(), valid_rows['True'].max()],
             color='red',
             linestyle='--',
             label='Perfect Prediction')

    plt.xlabel('True Death Frame')
    plt.ylabel('Predicted Death Frame')
    plt.title('Predicted vs. True Cell Death Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as)
    avg_error = np.sum(
        np.abs(valid_rows['True'] - valid_rows['Predicted'])) / len(
            valid_rows['True'])
    print(f"\nAverage error in cell death estimation is {avg_error} frames.\n")


def km_plot(hdf5_files: list[Path],
            labels: list[str],
            save_as: Path,
            time_steps: tuple[int] = (1, 1),
            initial_cells_only: bool = True,
            num_frames=None):
    """Plot the Kaplan Meier estimator for hdf5_file/s."""

    if isinstance(hdf5_files, Path):
        hdf5_files = [hdf5_files]

    cmap = plt.get_cmap('Set1')
    colours = [cmap(i) for i in range(len(hdf5_files))]

    def survival_estimator(d_is: np.ndarray, n_is: np.ndarray):
        """Estimate surival funciotn at each time t_i.
        args:
            d_is = number of events(death) at each time t_i
            n_is = number of samples still alive at each time t_i.
        returns:
            S [float]
        """
        return np.cumprod(1 - (d_is) / (n_is))

    for hdf5_file, label, colour, time_step in zip(hdf5_files, labels, colours,
                                                   time_steps):

        with h5py.File(hdf5_file, 'r') as f:
            cell_deaths_arr = f['Cells']['Phase']['CellDeath']
            max_frames = f['Cells']['Phase']['Area'].shape[0]
            this_num_frames = min(
                num_frames,
                max_frames) if num_frames is not None else max_frames

            print(num_frames)
            cell_deaths = cell_deaths_arr[0]

            # Load only 'Area' to compute start/end frames
            area_data = CellType('Phase').get_features_xr(
                f, features=['Area'])['Area'].transpose(
                    'Cell Index',
                    'Frame').values  # shape: (num_cells, num_frames)
            area_data = area_data[:, :this_num_frames]

            # Use numpy vectorization for speed
            not_nan_mask = ~np.isnan(area_data)
            # First valid frame (start frame)
            # Last valid frame (end frame)
            reversed_mask = not_nan_mask[:, ::-1]
            last_frames = area_data.shape[1] - 1 - reversed_mask.argmax(axis=1)
            if initial_cells_only:
                initial_cells_mask = f['Cells']['Phase']['First Frame'][
                    0] == 0.0
                cell_deaths = cell_deaths[initial_cells_mask]
                last_frames = last_frames[initial_cells_mask]
        ds, ns = np.zeros(shape=num_frames), np.zeros(shape=num_frames)
        for cell_death, last_frame in zip(cell_deaths, last_frames):
            if np.isnan(cell_death):
                ns[np.arange(0, last_frame).astype(int)] += 1
            else:
                death_frame = int(cell_death)
                if death_frame < this_num_frames:
                    # Valid death event
                    ns[np.arange(0, death_frame)] += 1
                    ds[death_frame] += 1
                else:
                    # Died after truncation → treat as censored
                    ns[np.arange(0, this_num_frames)] += 1
        mask = ds > 0
        ds = ds[mask]
        ns = ns[mask]
        ts = np.nonzero(mask)[0]
        ts = ts * time_step
        print(ts)
        print(mask.shape, ds.shape, ns.shape, ts.shape)
        plt.step(ts,
                 survival_estimator(ds, ns),
                 where='post',
                 label=label,
                 color=colour)

    plt.xlabel('Time / minutes')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid()
    plt.savefig(save_as)


def compare_cell_features(hdf5_files: list[Path], labels: list[str],
                          feature: str, save_as: Path) -> None:
    """Compare the histograms of a given cell feature between two datasets."""

    if isinstance(hdf5_files, Path):
        hdf5_files = [hdf5_files]

    cmap = plt.get_cmap('Set1')
    num_colours = len(hdf5_files)
    colours = [cmap(i) for i in range(len(hdf5_files))]

    for hdf5_file, label, colour in zip(hdf5_files, labels, colours):
        with h5py.File(hdf5_file, 'r') as f:
            features_ds = f['Cells']['Phase'][feature][:]
            cell_deaths_arr = f['Cells']['Phase']['CellDeath']
            num_frames = cell_deaths_arr.shape[0]

            print(num_frames)
            cell_deaths = cell_deaths_arr[0]
            for i, cell_death in enumerate(cell_deaths):
                if not np.isnan(cell_death):
                    features_ds[int(cell_death):, i] = np.nan

            plt.hist(features_ds.ravel(),
                     bins=50,
                     alpha=0.5,
                     color=colour,
                     label=label,
                     density=True)

    plt.xlabel(feature)
    plt.ylabel('frequency')
    plt.legend()
    plt.grid()
    plt.savefig(save_as)


def compare_cell_features_grid(
        hdf5_files: list[Path],
        labels: list[str],
        features: list[str],
        save_as: Path,
        alpha: float = 0.1  # significance level
) -> None:

    n_features = len(features)
    n_cols = 3  # histogram | KS | median ± 5–95%
    n_rows = n_features

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = plt.get_cmap('Set1')
    colours = [cmap(i) for i in range(len(hdf5_files))]

    # KS critical value constants
    c_alpha_values = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}
    c_alpha = c_alpha_values.get(alpha, 1.36)

    for row, feature in enumerate(features):
        print(f'Getting feature {feature}')
        ax_hist = axes[row, 0]
        ax_bar = axes[row, 1]
        ax_summary = axes[row, 2]

        all_data = []

        # --- Load and process feature data ---
        for hdf5_file, label, colour in tqdm(zip(hdf5_files, labels, colours)):
            with h5py.File(hdf5_file, 'r') as f:
                features_ds = f['Cells']['Phase'][
                    feature][:]  # shape: (frames, cells)
                # cell_deaths_arr = f['Cells']['Phase']['CellDeath']

                # # Mask values after cell death
                # cell_deaths = cell_deaths_arr[0]
                # for i, cell_death in enumerate(cell_deaths):
                #     if not np.isnan(cell_death):
                #         features_ds[int(cell_death):, i] = np.nan

                # Average per cell over time
                cell_means = features_ds.ravel()
                # cell_means = np.nanmean(features_ds, axis=0)
                cell_means = cell_means[~np.isnan(cell_means)]
                all_data.append(cell_means)

                # Histogram
                ax_hist.hist(cell_means,
                             bins=50,
                             alpha=0.5,
                             color=colour,
                             density=True,
                             label=label)

        # --- Histogram axis ---
        ax_hist.set_title(feature)
        ax_hist.set_xlabel(feature)
        ax_hist.set_ylabel('Normalised Frequency')
        # ax_hist.grid()
        if row == 0:
            ax_hist.legend()

        # --- KS statistics ---
        pair_labels = []
        ks_values = []
        p_values = []
        D_crit_values = []
        for (i, j) in combinations(range(len(all_data)), 2):
            ks_result = ks_2samp(all_data[i], all_data[j])
            n1, n2 = len(all_data[i]), len(all_data[j])
            D_crit = c_alpha * np.sqrt((n1 + n2) / (n1 * n2))

            pair_labels.append(f"{labels[i]} vs\n{labels[j]}")
            ks_values.append(ks_result.statistic)
            p_values.append(ks_result.pvalue)
            D_crit_values.append(D_crit)

        # sig_mask = np.array(p_values) < alpha
        sig_mask = np.zeros_like(p_values)

        bars = ax_bar.bar(range(len(ks_values)),
                          ks_values,
                          color='black',
                          edgecolor='black')
        ax_bar.set_xticks(range(len(ks_values)))
        ax_bar.set_xticklabels(pair_labels)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel('KS statistic')
        ax_bar.set_title(f"{feature} Pairwise KS")

        # Annotate bars and draw Dcrit lines
        for b, val, sig, D_crit in zip(bars, ks_values, sig_mask,
                                       D_crit_values):
            ax_bar.text(b.get_x() + b.get_width() / 2,
                        val + 0.02,
                        f"{val:.2f}" + (" *" if sig else ""),
                        ha='center',
                        va='bottom',
                        fontsize=10,
                        color='black')

            # ax_bar.hlines(D_crit, b.get_x(), b.get_x() + b.get_width(),
            #               colors='blue', linestyles='dashed', linewidth=1)

        # if row == 0:
        #     ax_bar.text(0.95, 0.95, f"* p < {alpha} (significant at α = {alpha})",
        #                 ha='right', va='top', transform=ax_bar.transAxes, color='black', fontsize=10)
        # ax_bar.text(0.95, 0.88, f"Blue dashed = D_crit",
        #             ha='right', va='top', transform=ax_bar.transAxes, color='blue', fontsize=9)

        # --- Median + 5–95th percentile summary ---
        # medians = [np.median(d) for d in all_data]
        # p5 = [np.percentile(d, 5) for d in all_data]
        # p95 = [np.percentile(d, 95) for d in all_data]
        # yerr = np.array([np.array(medians) - np.array(p5), np.array(p95) - np.array(medians)])

        # ax_summary.bar(range(len(labels)), medians, color=colours, alpha=1.0, edgecolor='none')
        bp = ax_summary.boxplot(all_data,
                                labels=labels,
                                patch_artist=True,
                                medianprops=dict(color='black'),
                                flierprops=dict(marker='.',
                                                markerfacecolor='black',
                                                markersize=2))
        for patch, colour in zip(bp['boxes'], colours):
            patch.set_facecolor(colour)

        # ax_summary.errorbar(range(len(labels)), medians, yerr=yerr, fmt='none', ecolor='black', capsize=5)
        ax_summary.set_xticks(range(1, len(labels) + 1))
        ax_summary.set_xticklabels(labels)
        ax_summary.set_ylabel(f"{feature}")
        ax_summary.set_title(f"{feature} Summary Stats")
        # ax_summary.grid(axis='y')

    plt.tight_layout()
    plt.savefig(save_as)
    plt.close(fig)


def plot_alive_vs_dead_feature(hdf5_file: Path, feature: str,
                               save_as: Path) -> None:
    """Plot histograms of a given feature for cells that survive vs cells that die."""

    with h5py.File(hdf5_file, 'r') as f:
        features_ds = f['Cells']['Phase'][
            feature][:]  # shape: (num_frames, num_cells)
        cell_deaths = f['Cells']['Phase']['CellDeath'][
            0]  # shape: (num_cells,)

        num_frames, num_cells = features_ds.shape

        alive_values = []
        dead_values = []

        for cell_idx in range(num_cells):
            death_frame = cell_deaths[cell_idx]

            if np.isnan(death_frame):
                # Alive — include all frames
                alive_values.append(features_ds[:, cell_idx])
            else:
                # Dead — include up to death frame (exclude death frame itself)
                dead_values.append(features_ds[:int(death_frame), cell_idx])

        # Flatten all values
        alive_values_flat = np.concatenate(
            alive_values) if alive_values else np.array([])
        dead_values_flat = np.concatenate(
            dead_values) if dead_values else np.array([])

        alive_values_flat = alive_values_flat[~np.isnan(alive_values_flat)]
        dead_values_flat = dead_values_flat[~np.isnan(dead_values_flat)]
        alive_values_flat = alive_values_flat[
            alive_values_flat <= np.percentile(alive_values_flat, 99)]
        dead_values_flat = dead_values_flat[dead_values_flat <= np.percentile(
            dead_values_flat, 99)]
        print(np.nanmax(alive_values_flat), np.nanmax(dead_values_flat))
        # Plot
        plt.hist(alive_values_flat,
                 bins=50,
                 alpha=0.6,
                 color='green',
                 label='Healthy cells',
                 density=True)
        plt.hist(dead_values_flat,
                 bins=50,
                 alpha=0.6,
                 color='red',
                 label='Dying cells',
                 density=True)

        plt.xlabel(feature)
        plt.ylabel('Frequency Density')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_as)


def plot_cell_positions(hdf5_file: Path,
                        title=None,
                        save_as: Path = None) -> None:
    with h5py.File(hdf5_file, 'r') as f:
        x = f['Cells']['Phase']['X'][:]  # shape: [frames, cells]
        y = f['Cells']['Phase']['Y'][:]

    # Flatten for plotting (all time points together)
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Optionally remove NaNs (cells missing in some frames)
    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]

    plt.figure(figsize=(6, 6))
    plt.scatter(x_flat, y_flat, s=10, alpha=0.6)
    plt.gca().invert_yaxis()  # optional if origin is top-left like images
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    if title is None:
        title = hdf5_file.stem
    plt.title(title)
    plt.axis('equal')
    plt.savefig(save_as)


def main():
    feature_names = [
        'Area',
        'Circularity',
        'Perimeter',
        'Speed',
        # 'Displacement',
        # 'Mode 0',
        # 'Mode 1',
        # 'Mode 2',
        # 'Mode 3',
        # 'Mode 4',
        # 'Speed',
        # 'Alive Phagocytes within 100 pixels',
        # 'Alive Phagocytes within 250 pixels',
        # 'Alive Phagocytes within 500 pixels',
        # 'Dead Phagocytes within 100 pixels',
        # 'Dead Phagocytes within 250 pixels',
        # 'Dead Phagocytes within 500 pixels',
        # 'Phagocytes within 100 pixels',
        # 'Phagocytes within 250 pixels',
        # 'Phagocytes within 500 pixels',

        # 'X',
        # 'Y',
        # 'CellDeath',
        # 'Total Fluorescence',
        # 'Fluorescence Distance Mean',
        # 'Fluorescence Distance Variance',
        # 'Inner Total Fluorescnece',
        # 'Outer Total FLuorescence',
        # 'External Fluorescence Intensity within 10 pixels',
        # 'External Fluorescence Intensity within 25 pixels',
        # 'External Fluorescence Intensity within 50 pixels',
        'Skeleton Length',
        # 'Skeleton Branch Points',
        # 'Skeleton End Points',
        # 'Skeleton Branch Length Mean',
        # 'Skeleton Branch Length Std',
        # 'Skeleton Branch Length Max',
    ]
    hdf5_files = [
        Path('~/thor_server/MacrophageData/24_07/').expanduser() / f'{_}.h5'
        for _ in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X')
    ]
    labels = (['No Fluor'] * 3 + ['Fluor'] * 6 + ['No Fluor'] * 6 +
              ['Fluor'] * 6 + ['No Fluor'] * 3)

    plot_per_well_summary(
        save_as=Path('temp') / 'per_well_summary.png',
        hdf5_files=hdf5_files,
        feature_names=feature_names,
        labels=labels,
    )
    plot_feature_correlations_multi(
        save_as=Path('temp') / 'features_comparison.png',
        hdf5_files=hdf5_files,
        feature_names=feature_names,
        labels=labels,
    )
    # plot_cell_features(Path('PhagoPred\\Datasets\\06_03\\H.h5'),
    #                    0,
    #                    0,
    #                    100,
    #                    Path('temp') / 'cell_features.png',
    #                    feature_names=feature_names)
    # plot_cell_tracks(Path('temp') / 'tracks.png',
    #                  50,
    #                  99,
    #                  99,
    #                  SETTINGS.DATASET,
    #                  linewidth=1.0)
    # plot_two_death_frame_hists(
    #     Path('temp') / 'death_frames_fine_tuned.txt',
    #     Path('temp') / 'cell_deaths_24_06.txt',
    #     label1='Fine Tuned',
    #     label2='Original',
    #     save_as=Path('temp') / 'death_frame_comparison.png'
    # )
    # plot_cell_features(1, 0, 50, Path('temp') / 'plot.png', feature_names=feature_names)
    # plot_average_cell_features(Path('temp') / 'plot.png', feature_names=feature_names)
    # plot_feature_correlations(Path('temp') / 'correlations_plot.png', feature_names=feature_names)
    # plot_num_dead_cells(Path('temp') / 'num_dead_cells_plot.png')
    # plot_feature_correlations(Path('temp') / 'plot.png', feature_names=[
    #     'Area',
    #     'Circularity',
    #     'Perimeter',
    #     'Displacement',
    #     'Mode 0',
    #     'Mode 1',
    #     'Mode 2',
    #     'Mode 3',
    #     'Speed'
    # ])

    # files = [
    #     Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '07_10_0.h5',
    #     Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '28_10_2500.h5',
    #     # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / 'old' / '03_10_2500.h5',
    #     Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000.h5',
    #     # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000_inner.h5',
    #     # Path('PhagoPred')/'Datasets'/ 'ExposureTest' / '10_10_5000_outer.h5',
    # ]
    # labels = [
    #     '0s exposure',
    #           '2.5s exposure',
    #         #   '2.5s (old data)',
    #           '5s exposure',
    #         #   'Inner radius',
    #         #   'Outer radius'
    #           ]
    # km_plot(files,
    #         labels,
    #         Path('temp') / 'km_curve.png',
    #         [5, 5, 5],
    #         num_frames = 750)
    # plot_num_alive_cells(files, Path('temp') / 'dead_cells.png', labels=labels, last_frame=600)
    # compare_cell_features(files,
    #                       labels,
    #                       'Speed',
    #                       Path('temp') / 'speed_plt.png')
    # compare_cell_features_grid(files,
    #                            labels,
    #                            features=feature_names,
    #                            save_as=Path('temp') / 'hists.png',
    #                            )
    # plot_feature_correlations_multi(
    #     Path('temp') / 'corr_plt.png',
    #     files,
    #     feature_names,
    #     labels,
    # )
    # plot_percentile_cell_features_multi(
    #     Path('temp') / 'features.png',
    #     files,
    #     0,
    #     750,
    #     labels=labels,
    #     feature_names=feature_names
    # )
    # # compare_cell_features([Path('PhagoPred') / 'Datasets' / '13_06_survival.h5',
    #             Path('PhagoPred') / 'Datasets' / '24_06_survival.h5'],
    #             ['13_06', '24_06'], 'Area',
    #             Path('temp') / 'features_plot.png')

    # plot_alive_vs_dead_feature(Path('PhagoPred') / 'Datasets' / '24_06_survival.h5', 'Circularity',Path('temp') / 'features_plot.png')
    # plot_cell_positions(files[1], save_as=Path('temp') / 'cell_positions_outer.png')


if __name__ == '__main__':
    main()
