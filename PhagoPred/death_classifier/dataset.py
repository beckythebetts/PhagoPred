from pathlib import Path
from typing import Optional, Sequence, Union
import json

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from PhagoPred.death_classifier.extract_crops import cache_name

# Channels which are 0-1 maps rather than images, so aren't normalised,
# intensity jittered or blurred like images
MAP_CHANNELS = ('mask', 'weight')


def load_labels(labels_path: Path) -> dict[str, Optional[int]]:
    """Labels from deaths.json, keyed by cache name."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return {
        cache_name(file, int(cell)): death_frame
        for file, cells in labels.items()
        for cell, death_frame in cells.items()
    }


def load_bad_tracks(labels_path: Path) -> list[str]:
    """Bad tracks, saved by label_gui.py as skipped cells, as cache names."""
    skipped_path = labels_path.with_name(f'{labels_path.stem}_skipped.json')
    if not skipped_path.exists():
        return []
    with open(skipped_path, 'r') as f:
        skipped = json.load(f)
    return [
        cache_name(file, int(cell)) for file, cells in skipped.items()
        for cell in cells
    ]


def unlabelled_cells(f: h5py.File,
                     name: str,
                     labels: dict,
                     skipped: dict,
                     min_track_length: int = 50) -> list[int]:
    """Cells in an h5 file with long enough tracks, not yet labelled.

    Args:
        f: open h5 file.
        name: its file name, as used in the labels files.
        labels: deaths.json contents (file -> {cell: death frame}).
        skipped: deaths_skipped.json contents (file -> [cells]).
        min_track_length: only cells tracked for at least this many frames
            (short tracks are mostly debris).
    """
    first_frames = f['Cells']['Phase']['First Frame'][0]
    last_frames = f['Cells']['Phase']['Last Frame'][0]
    track_lengths = np.nan_to_num(last_frames - first_frames)
    done = set(labels.get(name, {})) | set(
        str(cell) for cell in skipped.get(name, []))
    return [
        int(cell) for cell in np.nonzero(track_lengths >= min_track_length)[0]
        if str(cell) not in done
    ]


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own random state.

    Otherwise every worker copies the dataset's rng and samples identical
    windows.
    """
    # Augmentation uses torch on CPU, one thread per worker so workers
    # don't compete for the few CPUs
    torch.set_num_threads(1)
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def split_cells(labels: dict[str, Optional[int]],
                bad_keys: Sequence[str] = (),
                val_frac: float = 0.2,
                seed: int = 0) -> tuple[list[str], list[str]]:
    """Split cells into train / val, stratified by dead / alive / bad track.

    Split is by cell, not frame, as frames from one cell are highly
    correlated.
    """
    rng = np.random.default_rng(seed)
    bad_keys = set(bad_keys)
    groups = [
        [k for k, v in labels.items() if v is not None and k not in bad_keys],
        [k for k, v in labels.items() if v is None and k not in bad_keys],
        list(bad_keys),
    ]
    train, val = [], []
    for keys in groups:
        keys = sorted(keys)
        rng.shuffle(keys)
        num_val = int(round(val_frac * len(keys)))
        val += keys[:num_val]
        train += keys[num_val:]
    return train, val


def kfold_split(labels: dict[str, Optional[int]],
                bad_keys: Sequence[str] = (),
                n_folds: int = 5,
                fold: int = 0,
                seed: int = 0) -> tuple[list[str], list[str]]:
    """Train / val split for one fold of k-fold cross validation.

    Stratified by dead / alive / bad track, and by cell, so each cell is in
    the val set of exactly one fold. The fold assignment depends only on
    seed and the set of cells, so is the same for every fold of a run.
    """
    rng = np.random.default_rng(seed)
    bad_keys = set(bad_keys)
    groups = [
        [k for k, v in labels.items() if v is not None and k not in bad_keys],
        [k for k, v in labels.items() if v is None and k not in bad_keys],
        list(bad_keys),
    ]
    train, val = [], []
    for keys in groups:
        keys = sorted(keys)
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            (val if i % n_folds == fold else train).append(key)
    return train, val


class CellWindowDataset(Dataset):
    """Random windows of frames from labelled cells' crop sequences."""

    def __init__(self,
                 crops: Union[Path, dict[int, dict]],
                 labels: Optional[dict[str, Optional[int]]] = None,
                 keys: Optional[list[str]] = None,
                 bad_keys: Sequence[str] = (),
                 window: int = 64,
                 min_window: Optional[int] = None,
                 samples_per_cell: int = 10,
                 death_window_prob: float = 0.5,
                 channels: tuple[str, ...] = ('phase', 'mask'),
                 downsample: int = 1,
                 augment: bool = True,
                 strong_augment: bool = False,
                 background_dim: Optional[float] = None,
                 frame_diff: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            crops: directory of cached crops from extract_crops.py, or crops
                already in memory as returned by extract_crops.extract_file.
            labels: cache name -> death frame (None for alive cells). Only
                used for training targets, so not needed for inference
                (unlabelled cells are treated as alive).
            keys: cells to use, as cache names. Defaults to all labelled and
                bad track cells for a cache directory, or all cells for
                crops in memory.
            bad_keys: bad tracks, which may also be in keys. Their death
                labels (if any) are ignored.
            window: number of frames per sample. 0 for whole tracks, padded
                to the longest track, so training matches inference.
            min_window: if set, each sample is a crop of random length
                between min_window and the whole track (window is ignored),
                padded to the longest track. Gives the GRU sequences up to
                full length, as at inference, plus crops ending before death
                ("survives") and starting after it (dead from frame 0).
            samples_per_cell: windows per cell per epoch.
            death_window_prob: for dying cells, probability a window is
                chosen to contain the death frame (otherwise uniform).
            channels: which of the cache's channels (in each cell's json,
                e.g. phase, epi, mask) to use as input.
            downsample: spatial downsampling factor (by average pooling).
            augment: random flips, 90 degree rotations and phase intensity
                jitter, applied consistently across a window.
            strong_augment: if augmenting, also shift, scale, blur, add
                camera noise, use stronger intensity jitter, and randomly
                drop the mask channel (see _strong_augment).
            background_dim: if set, multiply image channels by
                background_dim + (1 - background_dim) * weight, using the
                cache's 'weight' channel (extract_crops.py --weight), so the
                background is dimmed to this fraction of its contrast while
                the cell and its surroundings are kept. The weight channel
                itself isn't an input.
            frame_diff: add a channel |frame t - frame t-1| for each image
                channel (after dimming and augmentation), so interior
                movement and its absence are explicit inputs.
            seed: random seed for window sampling and augmentation.
        """
        labels = {} if labels is None else labels
        self.bad_keys = set(bad_keys)
        if isinstance(crops, dict):
            tracks = {
                cache_name(t['meta']['file'], cell): t
                for cell, t in crops.items()
            }
            if keys is None:
                keys = sorted(tracks)
        else:
            cache_dir = crops
            if keys is None:
                keys = sorted(set(labels) | self.bad_keys)
            missing = [
                k for k in keys if not (cache_dir / f'{k}.npy').exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} labelled cells not in cache (run '
                    f'extract_crops.py first), e.g. {missing[:3]}')

        self.keys = keys
        self.labels = labels
        self.window = window
        self.min_window = min_window
        self.samples_per_cell = samples_per_cell
        self.death_window_prob = death_window_prob
        self.channels = tuple(channels)
        self.background_dim = background_dim
        self.frame_diff = frame_diff
        self.downsample = downsample
        self.augment = augment
        self.strong_augment = strong_augment
        self.rng = np.random.default_rng(seed)

        if isinstance(crops, dict):
            self.crops = {k: tracks[k]['crops'] for k in keys}
            self.meta = {k: tracks[k]['meta'] for k in keys}
        else:
            # Memory map crops so only the sampled window is read from disk
            self.crops = {
                k: np.load(cache_dir / f'{k}.npy', mmap_mode='r')
                for k in keys
            }
            self.meta = {}
            for k in keys:
                with open(cache_dir / f'{k}.json', 'r') as f:
                    self.meta[k] = json.load(f)
        layouts = {tuple(m['cache_channels']) for m in self.meta.values()}
        if len(layouts) > 1:
            raise ValueError(
                f'Cells have different channel layouts: {layouts}')
        cache_channels = layouts.pop()
        self.channel_idxs = [cache_channels.index(c) for c in channels]

        # Caches saved already downsampled (extract_crops.downsample_cache)
        # only need the remaining factor, and store masks x255
        cache_downsamples = {m.get('downsample', 1) for m in self.meta.values()}
        if len(cache_downsamples) > 1:
            raise ValueError(
                f'Cells are downsampled differently: {cache_downsamples}')
        cache_downsample = cache_downsamples.pop()
        if downsample % cache_downsample:
            raise ValueError(f'downsample {downsample} is not a multiple of '
                             f"the cache's {cache_downsample}")
        self.downsample = downsample // cache_downsample
        self.mask_idxs = [i for i, c in enumerate(channels) if c == 'mask'
                          ] if cache_downsample > 1 else []
        if background_dim is not None:
            if 'weight' not in cache_channels:
                raise ValueError('background_dim needs a cache with a weight '
                                 'channel (extract_crops.py --weight)')
            # Read the weight last, used for dimming then dropped
            self.channel_idxs.append(cache_channels.index('weight'))

        if self.window == 0 or self.min_window is not None:
            # Whole tracks or variable length crops, all padded to the
            # longest track so they can be batched
            self.window = max(len(c) for c in self.crops.values())

    def __len__(self) -> int:
        return len(self.keys) * self.samples_per_cell

    def _death_offset(self, key: str) -> Optional[int]:
        """Death frame relative to the start of the cached track.

        None for alive cells, and bad tracks which have no death label.
        """
        death_frame = None if key in self.bad_keys else self.labels.get(key)
        if death_frame is None:
            return None
        return max(death_frame - self.meta[key]['first_frame'], 0)

    def _sample_crop(self, key: str) -> tuple[int, int]:
        """Random (start, length) of a crop from a cell's track."""
        num_frames = len(self.crops[key])
        if self.min_window is None:
            length = self.window
        else:
            length = int(
                self.rng.integers(min(self.min_window, num_frames),
                                  num_frames + 1))
        max_start = max(num_frames - length, 0)
        death = self._death_offset(key)
        if death is not None and self.rng.random() < self.death_window_prob:
            # Crop containing the death frame
            low = max(death - length + 1, 0)
            high = min(death, max_start)
            return int(self.rng.integers(low, high + 1)), length
        return int(self.rng.integers(0, max_start + 1)), length

    def get_window(self,
                   key: str,
                   start: int,
                   length: Optional[int] = None,
                   pad_to: Optional[int] = None) -> dict:
        """Crop of a cell's frames and labels, padded to pad_to frames.

        Also used for inference on whole tracks (start=0,
        length=len(dataset.crops[key])).

        Args:
            key: cell.
            start: first frame of the crop, relative to the cached track.
            length: frames in the crop, defaults to window.
            pad_to: total length after padding, defaults to length.
        """
        length = self.window if length is None else length
        pad_to = length if pad_to is None else pad_to
        # With frame differences, also read the frame before the crop so the
        # first frame's difference is real rather than 0
        prev = 1 if self.frame_diff and start > 0 else 0
        crops = self.crops[key][start - prev:start + length, self.channel_idxs]
        num_valid = len(crops) - prev
        if self.downsample > 1:
            # Downsample first: normalisation, flips / rotations and intensity
            # jitter are linear so give the same result either way, and doing
            # them on 4x fewer pixels makes data loading (the training
            # bottleneck) much faster. Strong augmentation is applied at the
            # downsampled resolution
            # Mean of each d x d block by summing strided slices; much faster
            # than reshape(...).mean(), which was most of the loading time
            d = self.downsample
            crops = sum(crops[:, :, i::d, j::d].astype(np.float32)
                        for i in range(d) for j in range(d)) / (d * d)
        if self.mask_idxs:
            crops = crops.astype(np.float32)
            crops[:, self.mask_idxs] /= 255

        if self.background_dim is not None:
            weight = crops[:, -1:].astype(np.float32) / 255
            crops = crops[:, :-1]

        n = prev + num_valid  # frames read, including any previous frame
        is_image = np.array([c not in MAP_CHANNELS for c in self.channels])
        x = np.zeros((prev + pad_to, *crops.shape[1:]), dtype=np.float32)
        x[:n] = crops
        x = self._normalise(x)
        if self.background_dim is not None:
            # Normalised images are centred on 0 (mid grey), so this reduces
            # background contrast rather than darkening it
            x[:n, is_image] *= (self.background_dim +
                                (1 - self.background_dim) * weight)
        if self.augment:
            x = self._augment(x)
            if self.strong_augment:
                x[:n] = self._strong_augment(x[:n])
            # Intensity jitter shifts padding too, so re-zero it
            x[n:] = 0
        if self.frame_diff:
            # |frame t - frame t-1| of each image channel, after dimming and
            # augmentation so it matches the frames the model sees; makes
            # interior movement (or its absence after death) direct to read
            images = x[:, is_image]
            diff = np.zeros_like(images)
            diff[1:] = np.abs(images[1:] - images[:-1])
            diff[n:] = 0  # padding, including the jump into it
            x = np.concatenate([x, diff], axis=1)[prev:]

        valid = np.zeros(pad_to, dtype=bool)
        valid[:num_valid] = True

        death = self._death_offset(key)
        dead = np.zeros(pad_to, dtype=np.float32)
        if death is None or death - start >= num_valid:
            # No death within the frames shown (alive, or the crop ends
            # before the cell dies), so "survives", indexed after all frames
            death_idx = pad_to
        else:
            # 0 if the cell is already dead at the start of the crop
            death_idx = max(death - start, 0)
            dead[death_idx:num_valid] = 1
        # Dead at or before the crop's first frame (including cells labelled
        # dead at or before their track starts): no alive -> dead change is
        # visible, so with a dead_before output this is its own outcome
        # rather than death at frame 0
        dead_before = death is not None and death <= start

        return {
            # (C, T, H, W) for Conv3d
            # float16 halves the shared memory used passing batches from
            # DataLoader workers (a batch of 4 whole tracks is ~300 MB)
            'x':
            torch.from_numpy(
                np.ascontiguousarray(x.transpose(1, 0, 2, 3),
                                     dtype=np.float16)),
            'valid':
            torch.from_numpy(valid),
            'dead':
            torch.from_numpy(dead),
            'bad':
            torch.tensor(float(key in self.bad_keys)),
            'has_death_label':
            torch.tensor(key not in self.bad_keys),
            'death_idx':
            torch.tensor(death_idx),
            'dead_before':
            torch.tensor(dead_before),
            'key':
            key,
            'start':
            start,
        }

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        """Scale phase / epi to roughly [-1, 1], mask as 0 / 1, weight 0-1."""
        for i, channel in enumerate(self.channels):
            if channel == 'weight':
                x[:, i] = x[:, i] / 255
            elif channel != 'mask':
                x[:, i] = x[:, i] / 127.5 - 1
        return x

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Same random flip / rotation / intensity change for all frames."""
        x = np.rot90(x, k=int(self.rng.integers(4)), axes=(2, 3))
        if self.rng.random() < 0.5:
            x = x[:, :, :, ::-1]
        x = np.ascontiguousarray(x)
        jitter = 0.2 if self.strong_augment else 0.1
        for i, channel in enumerate(self.channels):
            if channel not in MAP_CHANNELS:
                x[:, i] = (x[:, i] * self.rng.uniform(1 - jitter, 1 + jitter) +
                           self.rng.uniform(-jitter, jitter))
        return x

    def _strong_augment(self, x: np.ndarray) -> np.ndarray:
        """Shift, scale, blur, camera noise and mask dropout.

        Shift, scale and blur are the same for every frame so they don't
        create fake movement between frames; noise is independent per frame,
        as real camera noise is.

        Args:
            x: (T, C, H, W) normalised valid frames.

        Returns:
            Augmented frames, same shape.
        """
        is_mask = np.array([c in MAP_CHANNELS for c in self.channels])
        image_channels = torch.from_numpy(np.nonzero(~is_mask)[0])
        mask_channels = torch.from_numpy(np.nonzero(is_mask)[0])
        generator = torch.Generator().manual_seed(int(self.rng.integers(
            2**31)))
        frames = torch.from_numpy(x)
        num_frames, _, height, _ = frames.shape

        # Zoom in by 1-1.2x and shift within the margin that leaves, as the
        # cell won't always be exactly centred or the same size. Only zoom in
        # and never shift past the crop edge, otherwise the padding has to
        # invent image (reflection gives unrealistic mirrored patterns)
        scale = self.rng.uniform(1.0, 1.2)
        max_shift = 1 - 1 / scale  # in grid units, where the crop is [-1, 1]
        shift = self.rng.uniform(-max_shift, max_shift, size=2)
        theta = torch.tensor(
            [[1 / scale, 0, shift[0]], [0, 1 / scale, shift[1]]],
            dtype=torch.float32)
        grid = F.affine_grid(theta.expand(num_frames, 2, 3),
                             frames.shape,
                             align_corners=False)
        frames = F.grid_sample(frames,
                               grid,
                               mode='bilinear',
                               padding_mode='reflection',
                               align_corners=False)
        frames[:, mask_channels] = (frames[:, mask_channels] > 0.5).float()

        images = frames[:, image_channels]
        # Blur, as from slightly different focus
        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.3, 1.2)
            radius = 3
            kernel = torch.exp(
                -0.5 * (torch.arange(-radius, radius + 1.0) / sigma)**2)
            kernel = kernel / kernel.sum()
            c = images.shape[1]
            images = F.conv2d(F.pad(images, (radius, radius, 0, 0),
                                    mode='reflect'),
                              kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1),
                              groups=c)
            images = F.conv2d(F.pad(images, (0, 0, radius, radius),
                                    mode='reflect'),
                              kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1),
                              groups=c)
        # Camera noise, independent per frame
        noise_sigma = self.rng.uniform(0, 0.06)
        images = images + noise_sigma * torch.randn(images.shape,
                                                    generator=generator)
        frames[:, image_channels] = images

        if len(mask_channels):
            if self.rng.random() < 0.1:
                # No mask for the whole window, so the model can't rely on it
                frames[:, mask_channels] = 0
            else:
                # Missing segmentations in single frames, as when tracking
                # drops detections
                dropped = torch.from_numpy(self.rng.random(num_frames) < 0.05)
                frames[dropped[:, None] & torch.isin(
                    torch.arange(frames.shape[1]), mask_channels)[None]] = 0
        return frames.numpy()

    def __getitem__(self, idx: int) -> dict:
        key = self.keys[idx % len(self.keys)]
        start, length = self._sample_crop(key)
        return self.get_window(key, start, length, pad_to=self.window)
