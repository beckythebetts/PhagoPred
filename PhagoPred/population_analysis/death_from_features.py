from pathlib import Path
from typing import Optional
import json
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Per-frame datasets which aren't useful as death indicators
EXCLUDE_FEATURES = ['X', 'Y', 'Macrophage']


def _per_cell_mean_abs_diff(values: np.ndarray, mask: np.ndarray,
                            min_frames: int) -> np.ndarray:
    """Mean absolute frame-to-frame change of each cell's values within mask.

    Only differences between consecutive frames which are both in mask are
    used, so changes across the death frame aren't counted in either state.

    Args:
        values: (frames, cells) feature array.
        mask: (frames, cells) frames to include.
        min_frames: cells with fewer valid differences than this are dropped.

    Returns:
        (cells,) mean |delta|, for cells with at least min_frames differences.
    """
    diffs = np.abs(np.diff(values, axis=0))
    diffs = np.where(mask[1:] & mask[:-1], diffs, np.nan)
    num_valid = np.sum(~np.isnan(diffs), axis=0)
    with warnings.catch_warnings():
        # Cells with no frames in this state give all-NaN columns
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_diffs = np.nanmean(diffs, axis=0)
    return mean_diffs[num_valid >= min_frames]


def _plot_cell_changes(features: list[str], alive_changes: dict,
                       dead_changes: dict, save_path: Path) -> None:
    """Bar of median per-cell mean |delta| for alive / dead, cells overlaid.

    Median across cells so a few broken tracks don't dominate.
    """
    rng = np.random.default_rng(0)
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axs = plt.subplots(nrows,
                            ncols,
                            figsize=(3 * ncols, 3 * nrows),
                            squeeze=False)

    for ax, feature in zip(axs.flat, features):
        for x, (changes, colour) in enumerate((
            (np.concatenate(alive_changes[feature]), 'tab:grey'),
            (np.concatenate(dead_changes[feature]), 'tab:red'),
        )):
            if len(changes) == 0:
                continue
            ax.bar(x,
                   np.median(changes),
                   color=colour,
                   alpha=0.5,
                   edgecolor='none')
            ax.scatter(x + rng.uniform(-0.2, 0.2, len(changes)),
                       changes,
                       color=colour,
                       s=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([
            f'Alive\n(n={len(np.concatenate(alive_changes[feature]))})',
            f'Dead\n(n={len(np.concatenate(dead_changes[feature]))})',
        ])
        ax.set_title(feature, fontsize=10)
        ax.grid(True, axis='y')

    for ax in axs.flat[len(features):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('Per-cell mean |Δ| per frame')

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)


def view_features(dataset_dir: Path,
                  save_path: Path = Path('temp') /
                  'death_featurecomaprison.png',
                  features: Optional[list[str]] = None,
                  bins: int = 40,
                  min_frames: int = 5) -> None:
    """Compare feature distributions for frames where cells are alive vs dead.

    deaths.json in dataset_dir maps h5 file name -> {cell idx: death frame},
    with null for cells which stay alive. Frames before a cell's death frame
    count as alive, frames from the death frame onwards count as dead, so
    cells which stay alive only contribute to the alive distribution.

    Also saves a second figure (save_path with '_cell_changes' appended) of
    each cell's mean absolute frame-to-frame change, computed separately for
    its alive and dead frames.

    Args:
        dataset_dir: directory containing deaths.json and the h5 files.
        save_path: where to save the figure.
        features: features to plot, defaults to all per-frame features.
        bins: number of histogram bins.
        min_frames: minimum frame-to-frame differences in a state for a
            cell's mean |delta| to be included.
    """
    deaths_json = dataset_dir / 'deaths.json'
    with open(deaths_json, 'r') as f:
        deaths_dict = json.load(f)

    alive_values = {}
    dead_values = {}
    alive_changes = {}
    dead_changes = {}
    for file, cells in deaths_dict.items():
        cell_idxs = np.array([int(cell) for cell in cells])
        death_frames = np.array(list(cells.values()), dtype=float)
        # Cells which stay alive never reach their death frame
        death_frames = np.where(np.isnan(death_frames), np.inf, death_frames)

        with h5py.File(dataset_dir / file, 'r') as f:
            group = f['Cells']['Phase']
            if features is None:
                features = [
                    name for name, ds in group.items()
                    if ds.shape[0] > 1 and name not in EXCLUDE_FEATURES
                ]
            # Read whole rows before indexing, h5py needs increasing indices
            first_frames = group['First Frame'][0][cell_idxs]
            last_frames = group['Last Frame'][0][cell_idxs]

            num_frames = group[features[0]].shape[0]
            frames = np.arange(num_frames)[:, np.newaxis]
            tracked = (frames >= first_frames) & (frames <= last_frames)
            dead = frames >= death_frames

            for feature in features:
                values = group[feature][:][:, cell_idxs]
                alive_values.setdefault(feature,
                                        []).append(values[tracked & ~dead])
                dead_values.setdefault(feature,
                                       []).append(values[tracked & dead])
                alive_changes.setdefault(feature, []).append(
                    _per_cell_mean_abs_diff(values, tracked & ~dead,
                                            min_frames))
                dead_changes.setdefault(feature, []).append(
                    _per_cell_mean_abs_diff(values, tracked & dead,
                                            min_frames))

    plt.rcParams["font.family"] = 'serif'
    ncols = 4
    nrows = int(np.ceil(len(features) / ncols))
    fig, axs = plt.subplots(nrows,
                            ncols,
                            figsize=(4 * ncols, 3 * nrows),
                            squeeze=False)

    for ax, feature in zip(axs.flat, features):
        alive = np.concatenate(alive_values[feature])
        dead = np.concatenate(dead_values[feature])
        alive = alive[~np.isnan(alive)]
        dead = dead[~np.isnan(dead)]

        # Shared bins so the two distributions are comparable
        edges = np.histogram_bin_edges(np.concatenate([alive, dead]),
                                       bins=bins)
        for values, state, colour in (
            (alive, 'Alive', 'tab:grey'),
            (dead, 'Dead', 'tab:red'),
        ):
            if len(values) == 0:
                continue
            ax.hist(
                values,
                bins=edges,
                density=True,
                alpha=0.5,
                color=colour,
                label=
                f'{state}: {np.mean(values):.3g} ± {np.std(values):.3g}\n(n={len(values)} frames)'
            )
        ax.set_title(feature, fontsize=10)
        ax.legend(fontsize='x-small')
        ax.grid(True)

    for ax in axs.flat[len(features):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150)

    _plot_cell_changes(
        features, alive_changes, dead_changes,
        save_path.with_name(
            f'{save_path.stem}_cell_changes{save_path.suffix}'))


if __name__ == '__main__':
    view_features(
        Path("~/thor_server").expanduser() / 'MacrophageData' / '02_10')
