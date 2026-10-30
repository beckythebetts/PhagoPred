"""Prototype cell death detection from frame-to-frame pixel change.

Live macrophages ruffle their membranes constantly while dead cells are
static, so the change in phase image intensity over a cell between frames
should drop when it dies. Unlike mask-based features, this doesn't depend on
the segmentation outline, so isn't swamped by mask jitter.
"""
from pathlib import Path
from typing import Optional
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


def compute_pixel_change(h5_path: Path,
                         cell_idxs: np.ndarray,
                         margin: int = 10) -> np.ndarray:
    """Normalised pixel change for each cell between consecutive frames.

    For frame t, the score is the mean |I_t - I_{t-1}| over the cell's masks
    at t and t-1, dilated by margin pixels and excluding other cells, divided
    by the mean |I_t - I_{t-1}| of the background (removes illumination
    flicker).

    Args:
        h5_path: dataset with Images/Phase and Segmentations/Phase.
        cell_idxs: cells to compute for.
        margin: pixels to dilate the cell masks by, so membrane ruffles at
            the edge are included.

    Returns:
        (frames, cells) scores, NaN where the cell isn't segmented in both
        frames (always NaN at frame 0).
    """
    cell_idxs = np.asarray(cell_idxs)
    with h5py.File(h5_path, 'r') as f:
        images = f['Images']['Phase']
        masks = f['Segmentations']['Phase']
        num_frames = images.shape[0]
        scores = np.full((num_frames, len(cell_idxs)), np.nan)

        prev_image = images[0].astype(np.float32)
        prev_mask = masks[0]
        # Background is -1, cell c is labelled c, so find_objects()[c] is c
        prev_objects = ndimage.find_objects(prev_mask.astype(np.int32) + 1)

        for t in tqdm(range(1, num_frames), desc=f'Pixel change {h5_path.name}'):
            image = images[t].astype(np.float32)
            mask = masks[t]
            objects = ndimage.find_objects(mask.astype(np.int32) + 1)

            # Subsample background for speed
            sub = (slice(None, None, 4), slice(None, None, 4))
            background = (mask[sub] == -1) & (prev_mask[sub] == -1)
            background_change = np.mean(
                np.abs(image[sub] - prev_image[sub])[background])

            for i, cell in enumerate(cell_idxs):
                now = objects[cell] if cell < len(objects) else None
                prev = prev_objects[cell] if cell < len(prev_objects) else None
                if now is None or prev is None:
                    continue
                crop = tuple(
                    slice(max(min(a.start, b.start) - margin, 0),
                          min(max(a.stop, b.stop) + margin, size))
                    for a, b, size in zip(now, prev, mask.shape))
                mask_now, mask_prev = mask[crop], prev_mask[crop]
                is_cell = (mask_now == cell) | (mask_prev == cell)
                region = ndimage.binary_dilation(is_cell, iterations=margin)
                is_other = (((mask_now != cell) & (mask_now != -1)) |
                            ((mask_prev != cell) & (mask_prev != -1)))
                region &= ~is_other
                change = np.abs(image[crop] - prev_image[crop])[region]
                scores[t, i] = np.mean(change) / background_change

            prev_image, prev_mask, prev_objects = image, mask, objects

    return scores


def predict_death(scores: np.ndarray,
                  threshold: float,
                  window: int = 5,
                  frac: float = 0.9,
                  min_dead: int = 10) -> tuple[Optional[int], np.ndarray]:
    """Predict a cell's death frame from its pixel change scores.

    Death is irreversible, so the death frame is the first frame after which
    the smoothed score stays below threshold (for at least frac of the
    remaining valid frames, and at least min_dead of them).

    Args:
        scores: (frames,) pixel change scores for one cell.
        threshold: score below which a cell counts as static.
        window: rolling median window (frames) for smoothing.
        frac: fraction of remaining frames which must be below threshold.
        min_dead: minimum number of below-threshold frames after death.

    Returns:
        Death frame (None if predicted alive), and the smoothed log scores.
    """
    smooth = pd.Series(np.log(scores)).rolling(
        window, center=True, min_periods=1).median().to_numpy()
    valid = ~np.isnan(smooth)
    below = valid & (smooth < np.log(threshold))
    # Counts of valid / below-threshold frames from each frame to the end
    valid_after = np.cumsum(valid[::-1])[::-1]
    below_after = np.cumsum(below[::-1])[::-1]
    dead_from = below & (below_after >= min_dead) & (below_after
                                                     >= frac * valid_after)
    death_frame = int(np.argmax(dead_from)) if dead_from.any() else None
    return death_frame, smooth


def _is_correct(predicted: Optional[int], true: Optional[int],
                tolerance: int) -> bool:
    if predicted is None or true is None:
        return predicted is None and true is None
    return abs(predicted - true) <= tolerance


def _best_threshold(cells: list[tuple[np.ndarray, Optional[int]]],
                    thresholds: np.ndarray, tolerance: int,
                    **kwargs) -> float:
    """Threshold which gets the most cells correct (median of ties)."""
    num_correct = np.array([
        sum(
            _is_correct(predict_death(scores, t, **kwargs)[0], true,
                        tolerance) for scores, true in cells)
        for t in thresholds
    ])
    return float(np.median(thresholds[num_correct == num_correct.max()]))


def evaluate(dataset_dir: Path,
             out_dir: Path = Path('temp') / 'pixel_change',
             tolerance: int = 5,
             recompute: bool = False,
             **kwargs) -> pd.DataFrame:
    """Test pixel change death detection on manually labelled cells.

    Threshold is chosen leave-one-cell-out: each cell is predicted using the
    threshold which does best on all the other cells.

    Args:
        dataset_dir: directory containing deaths.json and the h5 files.
        out_dir: where to save plots and cached scores.
        tolerance: predicted death within this many frames counts as correct.
        recompute: recompute scores even if cached.
        **kwargs: passed to predict_death (window, frac, min_dead).

    Returns:
        Per cell results.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_dir / 'deaths.json', 'r') as f:
        deaths_dict = json.load(f)

    names, cells = [], []
    for file, file_cells in deaths_dict.items():
        cell_idxs = np.array([int(cell) for cell in file_cells])
        cache = out_dir / f'scores_{Path(file).stem}.npz'
        if cache.exists() and not recompute:
            cached = np.load(cache)
            if set(cell_idxs) <= set(cached['cell_idxs']):
                lookup = {c: i for i, c in enumerate(cached['cell_idxs'])}
                scores = cached['scores'][:, [lookup[c] for c in cell_idxs]]
            else:
                cache.unlink()
        if not cache.exists():
            scores = compute_pixel_change(dataset_dir / file, cell_idxs)
            np.savez(cache, cell_idxs=cell_idxs, scores=scores)

        for i, (cell, death_frame) in enumerate(file_cells.items()):
            names.append(f'{Path(file).stem}/{cell}')
            cells.append((scores[:, i], death_frame))

    all_scores = np.concatenate([s for s, _ in cells])
    thresholds = np.exp(
        np.linspace(*np.log(np.nanpercentile(all_scores, [1, 99])), 200))

    rows = []
    for i, (name, (scores, true)) in enumerate(zip(names, cells)):
        others = cells[:i] + cells[i + 1:]
        threshold = _best_threshold(others, thresholds, tolerance, **kwargs)
        predicted, smooth = predict_death(scores, threshold, **kwargs)
        rows.append({
            'cell': name,
            'true': true,
            'predicted': predicted,
            'error': None if predicted is None or true is None else
            predicted - true,
            'correct': _is_correct(predicted, true, tolerance),
            'threshold': threshold,
            'smooth': smooth,
        })
    results = pd.DataFrame(rows)

    _plot_traces(results, out_dir / 'traces.png')
    _plot_distributions(results, out_dir / 'distributions.png')

    print(results.drop(columns='smooth').to_string(index=False))
    print(f"\n{results['correct'].sum()}/{len(results)} correct "
          f"(state right and death within ±{tolerance} frames)")
    return results


def _plot_traces(results: pd.DataFrame, save_path: Path) -> None:
    plt.rcParams["font.family"] = 'serif'
    ncols = 3
    nrows = int(np.ceil(len(results) / ncols))
    fig, axs = plt.subplots(nrows,
                            ncols,
                            figsize=(5 * ncols, 2.2 * nrows),
                            sharey=True,
                            squeeze=False)
    for ax, (_, row) in zip(axs.flat, results.iterrows()):
        ax.plot(row['smooth'], color='tab:blue', lw=0.9)
        ax.axhline(np.log(row['threshold']), color='grey', ls=':', lw=0.8)
        if pd.notna(row['true']):
            ax.axvline(row['true'], color='black', ls='--', lw=1,
                       label='Manual')
        if pd.notna(row['predicted']):
            ax.axvline(row['predicted'], color='tab:red', lw=1,
                       label='Predicted')
        ax.set_title(f"{row['cell']} {'✓' if row['correct'] else '✗'}",
                     fontsize=9,
                     color='black' if row['correct'] else 'tab:red')
        ax.grid(True)
    for ax in axs.flat[len(results):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('log pixel change')
    for ax in axs[-1]:
        ax.set_xlabel('Frame')
    axs[0, 0].legend(fontsize='x-small')
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)


def _plot_distributions(results: pd.DataFrame, save_path: Path) -> None:
    """Smoothed log scores for frames manually labelled alive vs dead."""
    alive, dead = [], []
    for _, row in results.iterrows():
        frames = np.arange(len(row['smooth']))
        true = np.inf if pd.isna(row['true']) else row['true']
        alive.append(row['smooth'][frames < true])
        dead.append(row['smooth'][frames >= true])
    alive, dead = np.concatenate(alive), np.concatenate(dead)
    alive, dead = alive[~np.isnan(alive)], dead[~np.isnan(dead)]

    plt.rcParams["font.family"] = 'serif'
    fig, ax = plt.subplots(figsize=(6, 4))
    edges = np.histogram_bin_edges(np.concatenate([alive, dead]), bins=50)
    ax.hist(alive, bins=edges, density=True, alpha=0.5, color='tab:grey',
            label=f'Alive ({len(alive)} frames)')
    ax.hist(dead, bins=edges, density=True, alpha=0.5, color='tab:red',
            label=f'Dead ({len(dead)} frames)')
    ax.set_xlabel('Smoothed log pixel change')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=120)


if __name__ == '__main__':
    evaluate(Path("~/thor_server").expanduser() / 'MacrophageData' / '02_10')
