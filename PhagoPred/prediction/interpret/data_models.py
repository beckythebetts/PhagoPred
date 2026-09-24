from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import h5py
import numpy as np


class BackgroundEnum(str, Enum):
    OBSERVATIONAL = 'observational'
    INTERVENTIONAL = 'interventional'


class ExplainerEnum(str, Enum):
    GROUND_TRUTH = 'ground_truth'
    KERNEL = 'kernel'


def _as_str(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def _next_sample_idx(f: h5py.File) -> int:
    """Smallest integer not already used as a top-level group name in ``f``."""
    existing = [int(k) for k in f.keys() if k.isdigit()]
    return max(existing) + 1 if existing else 0


@dataclass
class SHAPResult:
    explainer: ExplainerEnum
    background: BackgroundEnum
    temporal: np.ndarray | None = None
    feature: np.ndarray | None = None
    temporal_feature: np.ndarray | None = None
    segment_boundaries: np.ndarray | None = None


@dataclass
class SampleWithSHAP:
    feature_vals: np.ndarray
    feature_names: list[str]
    landmark_frame: int
    death_frame: int | None = None
    noise_vals: np.ndarray | None = None
    horizon_hazard: np.ndarray | None = None
    shap_vals: dict[tuple[ExplainerEnum, BackgroundEnum],
                    SHAPResult] = field(default_factory=dict)

    def explainers(self):
        return list([x[0] for x in self.shap_vals.keys()])

    def backgrounds(self):
        return list(x[1] for x in self.shap_vals.keys())

    def write_h5(self,
                 h5_file: h5py.File | Path | str,
                 sample_idx: int | None = None) -> int:
        """Write this sample, creating the file if it doesn't exist.

        """
        if isinstance(h5_file, (Path, str)):
            with h5py.File(h5_file, 'a') as f:
                return self._write_group(f, sample_idx)
        return self._write_group(h5_file, sample_idx)

    def _write_group(self, f: h5py.File, sample_idx: int | None) -> int:
        if sample_idx is None:
            sample_idx = _next_sample_idx(f)
        name = str(sample_idx)
        if name in f:
            del f[name]
        group = f.create_group(name)

        group.attrs['Features'] = self.feature_names
        group.attrs['Landmark Frame'] = self.landmark_frame
        if self.death_frame is not None:
            group.attrs['Death Frame'] = self.death_frame
        group.create_dataset('Signals', data=self.feature_vals, dtype=float)
        if self.noise_vals is not None:
            group.create_dataset('Noise', data=self.noise_vals, dtype=float)
        if self.horizon_hazard is not None:
            group.create_dataset('Horizon Hazard',
                                 data=self.horizon_hazard,
                                 dtype=float)

        for (explainer, background), result in self.shap_vals.items():
            tf = (result.temporal_feature
                  if result.temporal_feature is not None else np.zeros(0))
            ds = group.create_dataset(f'{explainer.value}__{background.value}',
                                      data=tf,
                                      dtype=float)
            ds.attrs['Explainer'] = explainer.value
            ds.attrs['Background'] = background.value
            if result.temporal is not None:
                ds.attrs['Temporal'] = result.temporal
            if result.feature is not None:
                ds.attrs['Feature'] = result.feature
            if result.segment_boundaries is not None:
                ds.attrs['Segment Boundaries'] = result.segment_boundaries

        return sample_idx

    @classmethod
    def read_h5(cls,
                h5_file: h5py.File | Path | str,
                sample_idx: int | None = None) -> 'SampleWithSHAP':
        """Read a sample back. ``sample_idx=None`` reads a single-sample file
        written at the root (no group nesting) — the counterpart to writing
        with an explicit index, not an auto-detect-the-latest-sample lookup.
        """
        if isinstance(h5_file, (Path, str)):
            with h5py.File(h5_file, 'r') as f:
                return cls._read_group(f, sample_idx)
        return cls._read_group(h5_file, sample_idx)

    @classmethod
    def _read_group(cls, f: h5py.File,
                    sample_idx: int | None) -> 'SampleWithSHAP':
        group = f[str(sample_idx)] if sample_idx is not None else f

        feature_names = [_as_str(n) for n in group.attrs['Features']]
        landmark_frame = int(group.attrs['Landmark Frame'])
        death_frame = group.attrs.get('Death Frame', None)
        death_frame = None if death_frame is None else float(death_frame)

        feature_vals = group['Signals'][:]
        noise_vals = group['Noise'][:] if 'Noise' in group else None
        horizon_hazard = (group['Horizon Hazard'][:]
                          if 'Horizon Hazard' in group else None)

        shap_vals = {}
        for name, ds in group.items():
            if name in ('Signals', 'Noise', 'Horizon Hazard'):
                continue
            explainer = ds.attrs.get('Explainer')
            if explainer is None:
                continue  # not a SHAPResult dataset
            explainer = ExplainerEnum(_as_str(explainer))
            background = BackgroundEnum(_as_str(ds.attrs['Background']))
            tf = ds[:]
            shap_vals[(explainer, background)] = SHAPResult(
                explainer=explainer,
                background=background,
                temporal_feature=tf if tf.size > 0 else None,
                temporal=(np.asarray(ds.attrs['Temporal'])
                          if 'Temporal' in ds.attrs else None),
                feature=(np.asarray(ds.attrs['Feature'])
                         if 'Feature' in ds.attrs else None),
                segment_boundaries=(np.asarray(
                    ds.attrs['Segment Boundaries'],
                    dtype=int) if 'Segment Boundaries' in ds.attrs else None),
            )

        return cls(
            feature_vals=feature_vals,
            feature_names=feature_names,
            landmark_frame=landmark_frame,
            death_frame=death_frame,
            noise_vals=noise_vals,
            horizon_hazard=horizon_hazard,
            shap_vals=shap_vals,
        )
