"""Per-well comparison of morphological features between groups of datasets."""

from typing import Optional, Union
import textwrap
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from scipy.stats import bootstrap
from tqdm import tqdm

plt.rcParams["font.family"] = 'serif'


def group_wells_by_label(
    hdf5_files: list[Union[str, Path]],
    labels: Optional[list[str]] = None,
) -> tuple[list[str], np.ndarray]:
    """Resolve repeated labels into groups of wells.

    Files sharing the same entry in ``labels`` form one group, e.g.
    ``labels=['No Fluor'] * 3 + ['Fluor'] * 6``. Returns the unique labels in
    the order they first appear, and the group index of each well.
    """
    if labels is None:
        labels = [Path(file_path).stem for file_path in hdf5_files]
    if len(labels) != len(hdf5_files):
        raise ValueError("Length of labels must match number of HDF5 files")

    group_labels = list(dict.fromkeys(labels))
    group_indices = np.array([group_labels.index(label) for label in labels])

    return group_labels, group_indices


def load_per_well_statistics(
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
) -> tuple[dict, dict, list[str]]:
    """Reduce each well to one median and one IQR per feature.

    The median and IQR are robust, so no outlier removal is applied. Returns
    ``(medians, iqrs, feature_names)``, each dict mapping a feature to one
    value per well in the order the files were given.
    """
    if feature_names is None:
        with h5py.File(hdf5_files[0], 'r') as f:
            feature_names = list(f['Cells']['Phase'].keys())

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

    return medians, iqrs, feature_names


def bootstrap_difference_ci(a: np.ndarray,
                            b: np.ndarray,
                            confidence: float = 0.95,
                            n_resamples: int = 9999,
                            seed: int = 0) -> tuple[float, float, float]:
    """Difference in means of ``b`` and ``a``, with a bootstrap interval.

    Bootstrapped rather than assuming normality, since there are only a
    handful of wells per group. Returns ``(difference, low, high)``.
    """
    a = np.asarray(a)[np.isfinite(a)]
    b = np.asarray(b)[np.isfinite(b)]
    difference = np.mean(b) - np.mean(a) if len(a) and len(b) else np.nan

    if len(a) < 2 or len(b) < 2:
        return difference, np.nan, np.nan

    def statistic(x, y, axis=-1):
        return np.mean(y, axis=axis) - np.mean(x, axis=axis)

    for method in ('BCa', 'percentile'):
        try:
            interval = bootstrap((a, b),
                                 statistic,
                                 method=method,
                                 confidence_level=confidence,
                                 n_resamples=n_resamples,
                                 random_state=seed).confidence_interval
            return difference, interval.low, interval.high
        except Exception:
            continue

    return difference, np.nan, np.nan


def plot_per_well_summary(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    seed: int = 0,
    annotate_wells: bool = False,
) -> plt.Figure:
    """One point per well (hdf5 file), grouped by label.

    ``annotate_wells`` labels each point with the stem of its hdf5 file, which
    is how an outlying well gets traced back to the dataset it came from.

    Treats the well, not the cell, as the unit of replication. Pooling millions
    of cells makes vanishingly small group differences look overwhelming, but
    cells within a well are not independent samples of "with fluorescence" -
    the wells are. The spread of the points within a group is the noise any
    real group difference has to beat, and it is readable directly off the
    plot.

    Top row is each well's median for a feature, bottom row its interquartile
    range (i.e. whether the spread differs, not just the centre). A box
    summarises each group, but the wells stay plotted on top of it: with ~12
    per group, the box's five numbers would hide the very spread the plot
    exists to show.
    """
    group_labels, group_indices = group_wells_by_label(hdf5_files, labels)
    num_groups = len(group_labels)
    well_names = [Path(file_path).stem for file_path in hdf5_files]

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(num_groups)]

    medians, iqrs, feature_names = load_per_well_statistics(
        hdf5_files, feature_names)

    num_features = len(feature_names)
    fig, axs = plt.subplots(2,
                            num_features,
                            figsize=(2.6 * num_features, 7),
                            squeeze=False)
    rng = np.random.default_rng(seed)

    for row, (statistic, values_by_feature) in enumerate([('Median', medians),
                                                          ('IQR', iqrs)]):
        for col, feat in enumerate(feature_names):
            ax = axs[row, col]
            values = np.array(values_by_feature[feat])
            # Track which well each point came from, so it can be labelled
            group_members = [
                np.flatnonzero((group_indices == k) & np.isfinite(values))
                for k in range(num_groups)
            ]
            group_values = [values[members] for members in group_members]

            # Box for the group, wells kept on top of it
            if all(len(y) > 0 for y in group_values):
                boxes = ax.boxplot(group_values,
                                   positions=range(num_groups),
                                   widths=0.5,
                                   showfliers=False,
                                   patch_artist=True,
                                   zorder=2)
                for k, patch in enumerate(boxes['boxes']):
                    # Alpha on the face only, so the border stays solid black
                    patch.set_facecolor(to_rgba(colors[k], 0.25))
                    patch.set_edgecolor('k')
                    patch.set_linewidth(2)
                for part in ('whiskers', 'caps', 'medians'):
                    for item in boxes[part]:
                        item.set_color('k')
                        item.set_linewidth(2)

            for k, y in enumerate(group_values):
                # Jittered strip of the individual wells
                x = k + rng.uniform(-0.12, 0.12, len(y))
                ax.scatter(x,
                           y,
                           color=colors[k],
                           s=35,
                           edgecolor='none',
                           zorder=3)

                if annotate_wells:
                    for x_i, y_i, well in zip(x, y, group_members[k]):
                        ax.annotate(well_names[well], (x_i, y_i),
                                    textcoords='offset points',
                                    xytext=(5, 0),
                                    va='center',
                                    fontsize=6,
                                    zorder=4)

            ax.set_xticks(range(num_groups))
            ax.set_xticklabels([
                f'{label}\n(n = {len(y)})'
                for label, y in zip(group_labels, group_values)
            ],
                               fontsize=10)
            ax.set_xlim(-0.6, num_groups - 0.4)
            ax.grid(visible=True, axis='y', which='both')

            if col == 0:
                ax.set_ylabel(f'Per-well {statistic}', fontsize=12)
            if row == 0:
                ax.set_title(textwrap.fill(feat, width=16), fontsize=12)

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
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_as, dpi=150, bbox_inches='tight')

    return fig


def plot_per_well_intervals(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    equivalence_percent: Optional[float] = None,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
) -> plt.Figure:
    """Group difference per feature, with a bootstrap confidence interval.

    The companion to plot_per_well_summary. That plot shows whether the groups
    look different; this one says how large a difference the data still allow,
    which is the question a non-significant test leaves open - failing to
    detect a difference is not evidence that there is none.

    Differences are shown as a percentage of the pooled value so that features
    on different scales sit on one axis. ``equivalence_percent`` optionally
    shades a band of differences small enough not to matter; choosing that
    bound is a scientific judgement, so there is no default.
    """
    group_labels, group_indices = group_wells_by_label(hdf5_files, labels)
    if len(group_labels) != 2:
        raise ValueError("Confidence intervals need exactly two groups, got "
                         f"{len(group_labels)}: {group_labels}")

    medians, iqrs, feature_names = load_per_well_statistics(
        hdf5_files, feature_names)

    num_features = len(feature_names)
    fig, ax = plt.subplots(figsize=(9, 0.62 * num_features + 2.6))

    for offset, (statistic, values_by_feature, marker,
                 color) in enumerate([('Median', medians, 'o', 'k'),
                                      ('IQR', iqrs, 's', 'grey')]):
        centres, lows, highs, positions = [], [], [], []

        for index, feat in enumerate(feature_names):
            values = np.array(values_by_feature[feat])
            difference, low, high = bootstrap_difference_ci(
                values[group_indices == 0],
                values[group_indices == 1],
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed)

            # Relative to the pooled value, so features are comparable
            scale = np.nanmean(values)
            if not np.isfinite([difference, low, high, scale]).all() or (scale
                                                                         == 0):
                continue

            centres.append(100 * difference / scale)
            lows.append(max(0.0, 100 * (difference - low) / scale))
            highs.append(max(0.0, 100 * (high - difference) / scale))
            positions.append(index + (offset - 0.5) * 0.3)

        ax.errorbar(centres,
                    positions,
                    xerr=[lows, highs],
                    fmt=marker,
                    color=color,
                    markersize=6,
                    capsize=4,
                    linewidth=1.4,
                    linestyle='None',
                    label=f'Per-well {statistic}')

    if equivalence_percent is not None:
        ax.axvspan(-equivalence_percent,
                   equivalence_percent,
                   color='grey',
                   alpha=0.15,
                   zorder=0,
                   label=f'±{equivalence_percent:g}% bound')
    ax.axvline(0, color='k', linewidth=1)

    ax.set_yticks(range(num_features))
    ax.set_yticklabels(feature_names, fontsize=11)
    ax.set_ylim(num_features - 0.5, -0.5)
    ax.set_xlabel(
        f'{group_labels[1]} - {group_labels[0]} '
        f'(% of pooled value, {confidence:.0%} bootstrap CI)',
        fontsize=11)
    ax.set_title(
        'An interval spanning 0 means no evidence of a difference, not '
        'evidence the groups match:\nits width is how large a difference the '
        'data still allow',
        fontsize=10,
        color='dimgrey')
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=10, frameon=False, loc='best')

    plt.tight_layout()
    plt.savefig(save_as, dpi=150, bbox_inches='tight')

    return fig


def main():
    feature_names = [
        'Area',
        'Circularity',
        'Perimeter',
        'Speed',
        'Skeleton Length',
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
        annotate_wells=False,
    )
    plot_per_well_intervals(
        save_as=Path('temp') / 'per_well_intervals.png',
        hdf5_files=hdf5_files,
        feature_names=feature_names,
        labels=labels,
    )


if __name__ == '__main__':
    main()
