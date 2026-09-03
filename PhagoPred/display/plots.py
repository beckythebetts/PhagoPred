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


def plot_per_well_summary(
    save_as: Path,
    hdf5_files: list[Union[str, Path]],
    feature_names: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    seed: int = 0,
    annotate_wells: bool = False,
) -> plt.Figure:
    """One point per well (hdf5 file), grouped by label.
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

    fig.suptitle('Per well feature comparison, day 3', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_as, dpi=150, bbox_inches='tight')

    return fig


def main():
    feature_names = [
        # 'Area',
        # 'Circularity',
        # 'Perimeter',
        # 'Speed',
        # 'Skeleton Length',
        'Area',
        'Circularity',
        'Displacement',
        'Perimeter',
        # 'Phagocytes within 100 pixels',
        # 'Phagocytes within 250 pixels',
        # 'Phagocytes within 500 pixels',
        # 'Skeleton Branch Length Mean',
        # 'Skeleton Branch Length Max',
        # 'Skeleton Branch Length Std',
        'Skeleton Branches',
        # 'Skeleton Length',
        # 'Speed',
        # 'Major Axis Length',
        # 'Minor Axis Length',
        'Eccentricity',
    ]
    hdf5_files = [
        Path('~/thor_server/MacrophageData/24_07/split_2').expanduser() /
        f'{_}.h5' for _ in ('A2', 'B2', 'C2', 'D2', 'E2', 'F2', 'G2', 'H2',
                            'I2', 'J2', 'K2', 'L2', 'M2', 'N2', 'O2', 'P2',
                            'Q2', 'R2', 'S2', 'T2', 'U2', 'V2', 'W2', 'X2')
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


if __name__ == '__main__':
    main()
