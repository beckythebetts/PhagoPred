"""DeathNet training config, saved with each run and checkpoint."""
from dataclasses import dataclass, fields, asdict
import json
from pathlib import Path

from PhagoPred.death_classifier.dataset import MAP_CHANNELS


@dataclass
class TrainConfig:
    labels_path: Path
    cache_dir: Path = Path('temp') / 'death_crops'
    out_dir: Path = Path('temp') / 'death_classifier'
    loss: str = 'frame'
    sigma: float = 2.0  # label smoothing (frames) for one_death loss
    # Weight of the bad track loss. 0 for a death only model, which also
    # leaves bad tracks out of training entirely
    bad_weight: float = 1.0
    bad_pool_r: float = 5.0  # smooth max sharpness for per-frame bad logits
    epochs: int = 50
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    window: int = 64  # 0 for whole tracks
    # If > 0, random length crops from min_window to whole tracks (window is
    # ignored); use for the death model, keep window 0 for the bad track model
    min_window: int = 0
    # For dying cells, probability a training crop is placed to contain the
    # death frame (otherwise uniform). Lower gives more "no death" and
    # "dead throughout" crops
    death_window_prob: float = 0.5
    samples_per_cell: int = 10
    channels: tuple[str, ...] = ('phase', 'mask')
    downsample: int = 1
    widths: tuple[int, ...] = (16, 32, 64, 128)
    hidden: int = 64
    dropout: float = 0.2
    val_frac: float = 0.2
    # If > 0, use fold `fold` of n_folds-fold cross validation instead of a
    # random val_frac split (see cross_validate.py)
    n_folds: int = 0
    fold: int = 0
    # Fraction of the training cells to use (stratified by dead / alive /
    # bad, nested so smaller fractions are subsets of larger), for learning
    # curves. Val cells are unchanged
    train_frac: float = 1.0
    # Evaluate on val cells every this many epochs (always on the last), as
    # whole-track evaluation is slow relative to training on few cells
    eval_every: int = 1
    tolerance: int = 5  # frames, for counting a death frame as correct
    num_workers: int = 2
    # Batches queued per worker; each whole track batch is ~300 MB of shared
    # memory, so keep low to avoid running out of RAM
    prefetch_factor: int = 1
    seed: int = 0
    amp: bool = True
    # Shifts, scaling, blur, noise and mask dropout on top of flips / rotations
    strong_augment: bool = False
    # Normalise CNN features per frame, so padding and sequence length don't
    # affect them (runs from before this option used False)
    frame_norm: bool = True
    # If > 0, dim the background to this fraction of its contrast using the
    # cache's weight channel (needs a cache from extract_crops.py --weight)
    background_dim: float = 0.0
    # Add |frame t - frame t-1| of each image channel as extra input channels
    frame_diff: bool = False
    # One-death loss only: add a separate "already dead when the sequence
    # starts" outcome, instead of counting it as death at frame 0
    dead_before: bool = False
    # How the cached crops were extracted, copied from the cache's json (same
    # names and units, see extract_crops.extract_file) when training starts
    # rather than given, so inference can make the same crops from h5 files
    crop_size: int = 256  # before downsampling
    smooth_window: int = 0
    cache_channels: tuple[str, ...] = ('phase', 'epi', 'mask')
    weight_grow: int = 20
    weight_fade: int = 20
    weight_temporal: int = 2
    _FROM_CACHE = ('crop_size', 'smooth_window', 'cache_channels',
                   'weight_grow', 'weight_fade', 'weight_temporal')

    # Options added after some runs were saved, with the value those runs
    # effectively used (frame_norm: plain GroupNorm before the option)
    _OLD_RUN_DEFAULTS = {
        'frame_norm': False,
        'background_dim': 0.0,
        'frame_diff': False,
        'dead_before': False,
    }

    def to_dict(self) -> dict:
        """JSON-safe dict: paths as strings, tuples as lists."""
        return {
            name:
            str(value) if isinstance(value, Path) else
            list(value) if isinstance(value, tuple) else value
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'TrainConfig':
        """Config from to_dict output or a checkpoint's 'config'.

        Restores paths and tuples, and options missing from configs saved
        before they existed get the value those runs used, not the current
        default.
        """
        config = {**cls._OLD_RUN_DEFAULTS, **config}
        for field in fields(cls):
            value = config.get(field.name)
            if value is None:
                continue
            if field.type is Path:
                config[field.name] = Path(value)
            elif getattr(field.type, '__origin__', None) is tuple:
                config[field.name] = tuple(value)
        return cls(**config)

    def to_json(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, path: Path) -> 'TrainConfig':
        """Config saved by to_json, e.g. a run's config.json."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def crop_kwargs(self) -> dict:
        """extract_file arguments that reproduce this model's training
        crops."""
        return dict(crop_size=self.crop_size,
                    smooth_window=self.smooth_window,
                    weight='weight' in self.cache_channels,
                    downsample=self.downsample,
                    weight_grow=self.weight_grow,
                    weight_fade=self.weight_fade,
                    weight_temporal=self.weight_temporal)

    def dataset_kwargs(self, training: bool = False) -> dict:
        """CellWindowDataset arguments for this model.
        """
        kwargs = dict(
            channels=self.channels,
            downsample=self.downsample,
            background_dim=self.background_dim or None,
            frame_diff=self.frame_diff,
            window=self.window,
            min_window=self.min_window or None,
            augment=training,
        )
        if training:
            kwargs.update(samples_per_cell=self.samples_per_cell,
                          death_window_prob=self.death_window_prob,
                          strong_augment=self.strong_augment,
                          seed=self.seed)
        return kwargs


def num_input_channels(cfg: TrainConfig) -> int:
    """Model input channels: chosen channels, plus a frame difference of
    each image (non map) channel if frame_diff."""
    images = [c for c in cfg.channels if c not in MAP_CHANNELS]
    return len(cfg.channels) + (len(images) if cfg.frame_diff else 0)
