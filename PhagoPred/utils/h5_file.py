from __future__ import annotations
import os

from pathlib import Path
from functools import cached_property
from typing import Iterable

import numpy as np
import h5py
from tqdm import tqdm


def _cell_feature(name: str) -> property:

    def getter(self: H5File) -> np.ndarray:
        return self.get_cell_feature(name)

    def setter(self, values: np.ndarray) -> None:
        self.write_feature(name, values)

    return property(getter, setter, doc=f"Per cell {name}")


def _attr(name: str) -> property:

    def getattr(self: H5File) -> object:
        return self.get_attr(name)

    def setattr(self: H5File, value: object) -> None:
        self.write_attr(name, value)

    return property(getattr, setattr)


def _ims_getter(path: str):

    def get(self: H5File,
            frames: int | list[int] | np.ndarray | slice | None = None,
            region: tuple[slice, slice] | None = None) -> np.ndarray:
        return self._get_ims(path, frames, region)

    get.__doc__ = (f'Frames of {path}: (H, W) for an int frame, else '
                   '(T, H, W); None for all frames. region=(rows, cols) '
                   'slices crops each frame.')
    return get


def _ims_writer(path: str, dtype: str, **default_create_kwargs):

    def write(self: H5File,
              batches: Iterable[np.ndarray],
              overwrite: bool = False,
              **create_kwargs) -> int:
        return self._write_ims(path, batches, dtype, overwrite, **{
            **default_create_kwargs,
            **create_kwargs
        })

    write.__doc__ = (f'Write {path} as {dtype} from (t, H, W) batches; see '
                     '_write_ims.')
    return write


class H5File:
    # === DATASET STRUCTURE ===
    EPI_THRESHOLD = 100

    FEATURES = 'Cells/Phase'
    FRAME_DIM, CELLS_DIM = 0, 1

    PHASE = 'Images/Phase'
    FLUOR = 'Images/Epi'
    SEGS = 'Segmentations/Phase'
    IM_FRAME_DIM = 0
    IM_SHAPE_DIMS = (1, 2)

    bad_track_thresh = 0.5

    # === Whole cell features ===
    first_frames = _cell_feature('First Frame')
    last_frames = _cell_feature('Last Frame')
    death_frames = _cell_feature('Death Frame', )
    event_frames = _cell_feature('EventDetection')

    # === Images / segmentations ===
    get_phase = _ims_getter(PHASE)
    get_segs = _ims_getter(SEGS)

    write_phase = _ims_writer(PHASE, 'uint8')
    write_epi = _ims_writer(FLUOR,
                            'uint16',
                            compression='gzip',
                            compression_opts=4)
    write_segs = _ims_writer(SEGS, 'int16')

    # === Dataset attributes ===
    camera = _attr('Camera')
    fov = _attr('FOV')
    frame_time = _attr('Time Interval / min')
    fluor_thresh = _attr('Fluor threshold')

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def _open(self, write: bool = False, create: bool = False) -> h5py.File:
        mode = 'a' if create else 'r+' if write else 'r'
        return h5py.File(self.path, mode)

    # === WHOLE DATASET VALUES ===

    @cached_property
    def num_cells(self) -> int:
        with self._open() as f:
            return f[self.FEATURES]['First Frame'].shape[self.CELLS_DIM]

    @cached_property
    def num_frames(self) -> int:
        with self._open() as f:
            return f[self.PHASE].shape[0]

    @cached_property
    def feature_names(self) -> list[str]:
        with self._open() as f:
            return [
                n for n, d in f[self.FEATURES].items()
                if d.shape[self.FRAME_DIM] > 1
            ]

    def ims_info(self, path: str) -> tuple[tuple, np.dtype] | None:
        """(shape, dtype) of an image stack, or None if it isn't in the file;
        no frames are read."""
        with self._open() as f:
            if path not in f:
                return None
            return f[path].shape, f[path].dtype

    @property
    def im_dims(self) -> tuple[int, int]:
        with self._open() as f:
            return tuple(f[self.PHASE].shape[d] for d in self.IM_SHAPE_DIMS)

    @property
    def good_cells(self) -> np.ndarray:
        if self.bad_track_thresh is None:
            return np.ones(self.num_cells, dtype=bool)
        else:
            return self.bad_track_prob <= self.bad_track_thresh

    # === READ/WRITE FEATURES ===

    @property
    def bad_track_prob(self) -> np.ndarray:
        """Never filtered: it's what the filter is based on."""
        with self._open() as f:
            return f[self.FEATURES]['Bad Track Probability'][0]

    @bad_track_prob.setter
    def bad_track_prob(self, array: np.ndarray) -> None:
        self.write_feature('Bad Track Probability', array)

    def get_features(
        self,
        names: str | list[str] | None = None,
        cell_idxs: int | list[int] | None = None,
        frames: int | list[int] | slice | None = None,
    ) -> dict | np.ndarray:
        """Per-frame features, (frames, cells); cells failing good_cells
        are NaN.

        Args:
            names: one feature, a list of them, or None for all of
                feature_names.
            cell_idxs, frames: index into the cell and frame axes; None for
                all. An int drops that axis.

        Returns:
            The array for a single name, else {name: array}.
        """
        single = isinstance(names, str)
        names = (self.feature_names
                 if names is None else [names] if single else list(names))
        cell_idxs = slice(None) if cell_idxs is None else cell_idxs
        frames = slice(None) if frames is None else frames
        # Features are chunked per cell, so int / slice indices are read
        # straight from the file (one cell is a single chunk); lists, which
        # h5py only partly supports, index a whole read instead
        direct = all(
            isinstance(i, (int, np.integer, slice))
            for i in (frames, cell_idxs))
        bad = ~self.good_cells[cell_idxs]
        values = {}
        with self._open() as f:
            for name in names:
                dataset = f[self.FEATURES][name]
                feature = (dataset[frames, cell_idxs] if direct else
                           dataset[()][frames, cell_idxs]).astype(float)
                # bad broadcasts over frames: cells are the last axis
                values[name] = np.where(bad, np.nan, feature)
        return values[names[0]] if single else values

    def write_feature(self, name: str, values: np.ndarray, **attrs) -> None:
        if values.ndim == 1:
            values = values[None, :]

        if values.shape[self.CELLS_DIM] != self.num_cells:
            raise ValueError(
                f'{name}: {values.shape[self.CELLS_DIM]} values for '
                f'{self.num_cells} cells')

        with self._open(write=True) as f:
            if name in f[self.FEATURES]:
                del f[self.FEATURES][name]
            dataset = f[self.FEATURES].create_dataset(name, data=values)
            dataset.attrs['dimensions'] = ['Frame', 'Cell Index']
            for k, v in attrs.items():
                dataset.attrs[k] = v

        self.reset_cache()

    # === READ/WRITE IMAGES/SEGMENATAIONS ===

    def _write_ims(
        self,
        path: str,
        batches: Iterable[np.ndarray],
        dtype: np.dtype | str,
        overwrite: bool = False,
        **create_kwargs,
    ) -> int:
        with self._open(create=True) as f:
            if path in f and not overwrite:
                raise FileExistsError(
                    f'{path} already in {self.path}; pass overwrite=True '
                    'to replace it')
            tmp_path = f'{path}_writing'
            if tmp_path in f:
                del f[tmp_path]

            dataset, num_frames = None, 0
            for batch in batches:
                batch = np.asarray(batch)
                if batch.ndim == 2:
                    batch = batch[None]
                if not np.can_cast(batch.dtype, dtype, 'safe'):
                    raise TypeError(
                        f'{path}: {batch.dtype} frames can\'t be stored as '
                        f'{np.dtype(dtype)} without losing values')
                if dataset is None:
                    frame_shape = batch.shape[1:]
                    dataset = f.create_dataset(tmp_path,
                                               shape=(0, *frame_shape),
                                               maxshape=(None, *frame_shape),
                                               dtype=dtype,
                                               chunks=(1, *frame_shape),
                                               **create_kwargs)
                dataset.resize(num_frames + len(batch), axis=self.IM_FRAME_DIM)
                dataset[num_frames:num_frames + len(batch)] = batch
                num_frames += len(batch)

            if dataset is None:
                raise ValueError(f'{path}: no frames to write')
            if path in f:
                del f[path]
            f.move(tmp_path, path)

        self.reset_cache()
        return num_frames

    def _get_ims(self,
                 path: str,
                 frames: int | list[int] | np.ndarray | slice | None = None,
                 region: tuple[slice, slice] | None = None) -> np.ndarray:
        if frames is None:
            frames = slice(None)
        rows, cols = (slice(None), slice(None)) if region is None else region
        with self._open() as f:
            if isinstance(frames, (int, np.integer, slice)):
                return f[path][frames, rows, cols]
            # h5py only takes increasing, unique index lists
            unique, inverse = np.unique(frames, return_inverse=True)
            return f[path][unique, rows, cols][inverse]

    def get_epi(self,
                frames: int | list[int] | np.ndarray | slice | None = None,
                region: tuple[slice, slice] | None = None,
                threshold: bool = False) -> np.ndarray:
        ims = self._get_ims(self.FLUOR, frames, region)
        if threshold:
            ims[ims < self.EPI_THRESHOLD] = 0
        return ims

    # === UTILS ===

    def reset_cache(self) -> None:
        """Forget cached properties, so they're re-read from the file."""
        for name, attr in vars(type(self)).items():
            if isinstance(attr, cached_property):
                self.__dict__.pop(name, None)

    def get_cell_feature(
        self,
        name: str,
    ) -> np.ndarray:
        """A per-cell dataset; cells failing good_cells are NaN."""
        with self._open() as f:
            values = f[self.FEATURES][name][0].astype(float)
        values[~self.good_cells] = np.nan
        return values

    def get_attr(self, name: str) -> object:
        with self._open() as f:
            return f.attrs[name]

    def write_attr(self, name: str, value: object) -> None:
        with self._open(write=True) as f:
            f.attrs[name] = value

    def delete_attr(self, name: str) -> None:
        with self._open(write=True) as f:
            if name in f.attrs:
                del f.attrs[name]

    @property
    def per_frame_fluorescence(self) -> np.ndarray:
        """Summed thresholded Epi intensity per frame, computed once and saved in the file."""
        with self._open() as f:
            if 'Total Fluorescence' in f.attrs:
                return f.attrs['Total Fluorescence']

        totals = np.array([
            self.get_epi(i, threshold=True).sum(dtype=np.uint64)
            for i in tqdm(range(self.num_frames),
                          desc='Computing total fluorescnce')
        ])

        self.write_attr('Total Fluorescence', totals)
        self.fluor_thresh = self.EPI_THRESHOLD
        return totals

    def repack(self) -> None:
        """Rewrite the file."""
        tmp = self.path.with_suffix('.repacking.h5')
        with self._open() as src, h5py.File(tmp, 'w') as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for name in src:
                src.copy(src[name], dst, name=name)
        os.replace(tmp, self.path)
        self.reset_cache()
