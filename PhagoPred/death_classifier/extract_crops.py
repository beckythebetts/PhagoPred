from __future__ import annotations

from pathlib import Path
import argparse
import json

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from PhagoPred.utils import tools

CHANNELS = ('phase', 'epi', 'mask')
# Alternative layout, with a background dimming weight instead of the mask
WEIGHT_CHANNELS = ('phase', 'weight')


def cache_name(file: str, cell: int) -> str:
    return f'{Path(file).stem}_{cell}'


def _crop_window(centre: tuple[float, float], crop_size: int,
                 image_shape: tuple[int, int]) -> tuple[tuple, tuple]:
    """Source slices in the image and destination slices in the crop.

    Crops overlapping the image edge are zero padded rather than shrunk, so
    every crop is crop_size x crop_size with the cell at the centre.
    """
    src, dst = [], []
    for c, size in zip(centre, image_shape):
        start = int(round(c)) - crop_size // 2
        end = start + crop_size
        src.append(slice(max(start, 0), min(end, size)))
        dst.append(
            slice(max(start, 0) - start, crop_size - (end - min(end, size))))
    return tuple(src), tuple(dst)


def _block_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks of the last two axes, as float32.

    Sums strided slices, as in CellWindowDataset, which is much faster than
    reshape(...).mean().
    """
    return sum(x[..., i::factor, j::factor].astype(np.float32)
               for i in range(factor) for j in range(factor)) / factor**2


def background_weight(masks: np.ndarray,
                      grow: int = 20,
                      fade: int = 20,
                      temporal: int = 2) -> np.ndarray:
    """Weight for dimming the background around a cell, from its masks.

    1 within grow px of the cell's mask, fading linearly to 0 over the next
    fade px. Each frame's weight is the max over +-temporal frames, so a
    missing or wrong mask in single frames doesn't dim the cell, and frames
    with no mask nearby get weight 1 everywhere (not dimmed) rather than
    hiding the cell.

    Args:
        masks: (T, H, W) bool masks of the cell.
        grow: pixels around the mask kept at full weight.
        fade: pixels over which the weight fades to 0 beyond that.
        temporal: frames either side to take the max over.

    Returns:
        (T, H, W) uint8 weights, 255 = full weight.
    """
    weights = np.zeros(masks.shape, dtype=np.float32)
    for t in np.nonzero(masks.reshape(len(masks), -1).any(axis=1))[0]:
        distance = ndimage.distance_transform_edt(~masks[t])
        weights[t] = np.clip(1 - (distance - grow) / fade, 0, 1)
    weights = ndimage.maximum_filter1d(weights,
                                       size=2 * temporal + 1,
                                       axis=0,
                                       mode='nearest')
    weights[~weights.reshape(len(weights), -1).any(axis=1)] = 1
    return np.round(weights * 255).astype(np.uint8)


def extract_file(
    h5_path: Path,
    cells: list[int] | None = None,
    cache_dir: Path | None = None,
    crop_size: int = 256,
    smooth_window: int = 0,
    weight: bool = False,
    downsample: int = 1,
    weight_grow: int = 20,
    weight_fade: int = 20,
    weight_temporal: int = 2,
) -> dict[int, dict]:
    """Extract crops for cells in one h5 file, reading each frame once.

    Args:
        h5_path: dataset to extract from.
        cells: cell indices to extract, defaults to all tracked cells.
        cache_dir: where to save crops, or None to only return them.
        crop_size: side length of square crops (pixels), before
            downsampling.
        smooth_window: if > 1, centre crops on a running median of the
            cell's position over this many frames, so the crop doesn't jump
            with segmentation changes or centroid jitter (whole crop shifts
            look like movement everywhere).
        weight: save channels (phase, weight), with weight from
            background_weight for dimming the background, instead of
            (phase, epi, mask). Padding of crops overlapping the image edge
            is filled with the median intensity rather than 0.
        downsample: if > 1, downsample each frame's crop by this factor
            (block means) as it is read, so only small crops are held in
            memory. Masks are saved as fractions x255, as in
            downsample_cache, and weights are computed from the downsampled
            masks with grow and fade scaled to match.
        weight_grow, weight_fade, weight_temporal: background_weight's
            grow, fade (full resolution pixels) and temporal (frames).

    Returns:
        cell -> {'first', 'last', 'meta', 'rows', 'cols', 'crops'}, with
        crops (T, C, crop_size // downsample, crop_size // downsample) uint8
        in the channel order above and meta as saved in the cache's json.
        Can be given directly to CellWindowDataset instead of a cache
        directory.
    """
    channels = WEIGHT_CHANNELS if weight else CHANNELS
    size = crop_size // downsample
    with h5py.File(h5_path, 'r') as f:
        phase, epi = f['Images']['Phase'], f['Images']['Epi']
        masks = f['Segmentations']['Phase']
        image_shape = phase.shape[1:]
        first_frames = f['Cells']['Phase']['First Frame'][0]
        last_frames = f['Cells']['Phase']['Last Frame'][0]
        # 'X' is the row centroid and 'Y' the column
        rows = f['Cells']['Phase']['X'][:]
        cols = f['Cells']['Phase']['Y'][:]
        if cells is None:
            cells = [int(c) for c in np.nonzero(~np.isnan(first_frames))[0]]

        tracks = {}
        for cell in cells:
            first, last = int(first_frames[cell]), int(last_frames[cell])
            num_frames = last - first + 1
            # Fill frames where the cell wasn't segmented with the
            # nearest known position
            track_rows = tools.fill_nans(rows[first:last + 1, cell])
            track_cols = tools.fill_nans(cols[first:last + 1, cell])
            if smooth_window > 1:
                track_rows, track_cols = (pd.Series(p).rolling(
                    smooth_window, center=True,
                    min_periods=1).median().to_numpy()
                                          for p in (track_rows, track_cols))
            tracks[cell] = {
                'first':
                first,
                'last':
                last,
                # As saved with cached crops
                # Crop settings use TrainConfig's names and units
                # (TrainConfig._FROM_CACHE), so train() copies them straight
                # into the config and inference passes them back here
                'meta': {
                    'file': h5_path.name,
                    'cell': cell,
                    'first_frame': first,
                    'last_frame': last,
                    'crop_size': crop_size,  # before downsampling
                    'cache_channels': channels,
                    'smooth_window': smooth_window,
                    'downsample': downsample,
                    'weight_grow': weight_grow,
                    'weight_fade': weight_fade,
                    'weight_temporal': weight_temporal,
                },
                'rows':
                track_rows,
                'cols':
                track_cols,
                'crops':
                np.zeros((num_frames, len(channels), size, size),
                         dtype=np.uint8),
                # Masks kept separately when computing weights from them
                'masks':
                np.zeros(
                    (num_frames, size, size), dtype=bool) if weight else None,
            }
        # Full resolution crop of one cell in one frame, before downsampling
        full = np.zeros((len(channels), crop_size, crop_size), dtype=np.uint8)
        full_mask = np.zeros((crop_size, crop_size), dtype=bool)

        start = min(t['first'] for t in tracks.values())
        end = max(t['last'] for t in tracks.values())
        for frame in tqdm(range(start, end + 1),
                          desc=f'Extracting {h5_path.name}'):
            active = [
                c for c, t in tracks.items()
                if t['first'] <= frame <= t['last']
            ]
            if not active:
                continue
            # Masks are compressed in whole frame chunks, so read the frame
            # once; images are uncompressed so crops can be read directly
            mask = masks[frame]
            for cell in active:
                track = tracks[cell]
                i = frame - track['first']
                src, dst = _crop_window((track['rows'][i], track['cols'][i]),
                                        crop_size, image_shape)
                crops = track['crops'][i] if downsample == 1 else full
                crop_mask = track['masks'][i] if (
                    weight and downsample == 1) else full_mask
                crops[:] = 0
                crop_mask[:] = False
                patch = phase[(frame, *src)]
                if weight and patch.shape != crops.shape[1:]:
                    # Crop overlaps the image edge: fill the padding with the
                    # median intensity rather than black, so the edge isn't a
                    # high contrast feature which moves with the crop
                    crops[0] = np.median(patch)
                crops[(0, *dst)] = patch
                if weight:
                    crop_mask[dst] = mask[src] == cell
                else:
                    crops[(1, *dst)] = epi[(frame, *src)]
                    crops[(2, *dst)] = mask[src] == cell
                if downsample > 1:
                    small = _block_mean(crops, downsample)
                    if not weight:
                        small[2] *= 255
                    track['crops'][i] = np.round(small)
                    if weight:
                        # Any of the block, so thin or small parts of the
                        # mask aren't lost (losing them drops their whole
                        # grow + fade region from the weight)
                        track['masks'][i] = _block_mean(crop_mask,
                                                        downsample) > 0
    if weight:
        for track in tracks.values():
            track['crops'][:, 1] = background_weight(track.pop('masks'),
                                                     weight_grow // downsample,
                                                     weight_fade // downsample,
                                                     weight_temporal)
    if cache_dir is not None:
        for cell, track in tqdm(tracks.items(), desc='Saving'):
            name = cache_name(h5_path.name, cell)
            # Save via temporary files so an interrupted run can't leave a
            # partial crop file that looks complete
            tmp_path = cache_dir / f'{name}.tmp.npy'
            np.save(tmp_path, track['crops'])
            tmp_path.replace(cache_dir / f'{name}.npy')
            with open(cache_dir / f'{name}.json', 'w') as f:
                json.dump(track['meta'], f)
    return tracks


def downsample_cache(src_dir: Path,
                     dst_dir: Path,
                     keys: list[str],
                     factor: int = 2,
                     channels: tuple[str, ...] = ('phase', 'mask'),
                     chunk: int = 32) -> None:
    """Copy cached crops at lower resolution, with only some channels.

    Training on whole tracks reads every cached frame each epoch, so a
    smaller cache that fits in RAM is much faster than downsampling on the
    fly. Pixels are block means as in CellWindowDataset. Masks become
    fractions at block edges, saved x255 to stay uint8 (CellWindowDataset
    rescales them). The json records the factor, so a dataset with
    downsample=factor uses the crops as they are.

    Args:
        src_dir: full resolution cache.
        dst_dir: where to save the downsampled cache.
        keys: cells to copy (cache names); already copied cells are skipped.
        factor: downsampling factor.
        channels: channels of the source cache to keep.
        chunk: frames processed at once, to bound memory.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for key in tqdm(keys, desc='Downsampling'):
        if (dst_dir / f'{key}.npy').exists():
            continue
        with open(src_dir / f'{key}.json', 'r') as f:
            meta = json.load(f)
        idxs = [meta['cache_channels'].index(c) for c in channels]
        crops = np.load(src_dir / f'{key}.npy', mmap_mode='r')
        num_frames, _, height, width = crops.shape
        out = np.empty(
            (num_frames, len(channels), height // factor, width // factor),
            dtype=np.uint8)
        for start in range(0, num_frames, chunk):
            mean = _block_mean(crops[start:start + chunk, idxs], factor)
            for c, channel in enumerate(channels):
                if channel == 'mask':
                    mean[:, c] *= 255
            out[start:start + chunk] = np.round(mean)
        tmp_path = dst_dir / f'{key}.tmp.npy'
        np.save(tmp_path, out)
        tmp_path.replace(dst_dir / f'{key}.npy')
        with open(dst_dir / f'{key}.json', 'w') as f:
            # crop_size stays the size before downsampling
            json.dump(
                {
                    **meta, 'cache_channels': channels,
                    'downsample': meta.get('downsample', 1) * factor
                }, f)


def extract_labelled(dataset_dir: Path,
                     cache_dir: Path,
                     labels_name: str = 'deaths.json',
                     crop_size: int = 256,
                     smooth_window: int = 0,
                     weight: bool = False,
                     cells_per_pass: int = 25,
                     downsample: int = 1) -> None:
    """Extract crops for all labelled and bad track cells not yet cached.

    Bad tracks are read from the skipped labels file saved by label_gui.py.
    With downsample > 1, cells_per_pass can be ~downsample**2 times larger.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_path = dataset_dir / labels_name
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    skipped_path = labels_path.with_name(f'{labels_path.stem}_skipped.json')
    skipped = {}
    if skipped_path.exists():
        with open(skipped_path, 'r') as f:
            skipped = json.load(f)

    files = set(labels) | set(skipped)
    for file in sorted(files):
        cells = set(int(c) for c in labels.get(file, {})) | set(
            int(c) for c in skipped.get(file, []))
        to_extract = sorted(
            cell for cell in cells
            if not (cache_dir / f'{cache_name(file, cell)}.npy').exists())
        if not to_extract:
            continue
        if not (dataset_dir / file).exists():
            print(f'{file} not found in {dataset_dir}, skipping '
                  f'{len(to_extract)} uncached cells')
            continue
        # A file's cells are held in memory until saved, so extract in
        # groups to keep memory bounded (~1.5 GB per 25 cells)
        for i in range(0, len(to_extract), cells_per_pass):
            extract_file(dataset_dir / file, to_extract[i:i + cells_per_pass],
                         cache_dir, crop_size, smooth_window, weight,
                         downsample)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract crops of labelled cells to a local cache.')
    parser.add_argument('dataset_dir',
                        type=Path,
                        help='Directory containing the .h5 files.')
    parser.add_argument('--cache-dir',
                        type=Path,
                        default=Path('temp') / 'death_crops')
    parser.add_argument('--labels', default='deaths.json')
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--smooth-window',
                        type=int,
                        default=0,
                        help='Frames for running median of crop centres.')
    parser.add_argument('--weight',
                        action='store_true',
                        help='Save (phase, weight) for background dimming '
                        'instead of (phase, epi, mask).')
    parser.add_argument('--downsample',
                        type=int,
                        default=1,
                        help='Downsample crops by this factor as extracted.')
    parser.add_argument('--cells-per-pass', type=int, default=25)
    args = parser.parse_args()
    extract_labelled(args.dataset_dir.expanduser(), args.cache_dir,
                     args.labels, args.crop_size, args.smooth_window,
                     args.weight, args.cells_per_pass, args.downsample)
