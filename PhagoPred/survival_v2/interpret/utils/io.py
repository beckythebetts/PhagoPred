from __future__ import annotations
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field, fields, Field

import numpy as np
import h5py
import json

from ...configs import PredictionTarget
from ...configs.io import to_json_safe, load_prediction_target
from ..data_models import SampleWithSHAP, SampleValues, SHAPResult
from ..config import InterpretCfg, load_interpret_cfg


def _next_sample_idx(f: h5py.Group) -> int:
    existing_idxs = [int(k) for k in f.keys() if k.isdigit()]
    return max(existing_idxs) + 1 if existing_idxs else 0


@dataclass
class ResolvedValues:
    background_set_id: str
    var_lag: int | None = None
    var_burn_in: int | None = None
    saved_burn_in_frames: int | None = None
    num_shap_samples: dict[str, int] | None = None


@dataclass
class SHAPFileMetadata:
    interpret: InterpretCfg
    prediction_target: PredictionTarget
    scenario: str | None = None

    resolved: ResolvedValues | None = field(default=None, compare=False)
    num_samples: int = field(default=0, compare=False)

    def can_reuse(self, requested: SHAPFileMetadata) -> bool:
        if self.resolved is None or requested.resolved is None:
            return False
        return (self == requested and self.resolved.background_set_id
                == requested.resolved.background_set_id and self.num_samples
                >= requested.interpret.selection.num_samples)


def _kind(f_: Field) -> str:
    try:
        return f_.metadata['kind']
    except KeyError:
        return 'dataset'


def _enum(f_: Field) -> type[Enum] | None:
    try:
        return f_.metadata['enum']
    except KeyError:
        return None


def write_sample(f: h5py.Group,
                 sample: SampleWithSHAP,
                 idx: int | None = None) -> int:
    if idx is None:
        idx = _next_sample_idx(f)

    if str(idx) in f:
        del f[str(idx)]

    group = f.create_group(str(idx))

    _write_fields(group, sample)

    return idx


def _write_fields(h5_group: h5py.Group, sample: SampleWithSHAP) -> None:
    for f_ in fields(sample):
        name = f_.name
        kind, value = _kind(f_), getattr(sample, name)
        if value is None:
            continue

        if name == 'feature_vals':
            _write_fields(h5_group, value)
        elif name == 'shap_vals':
            for (scorer, background), shap in value.items():
                shap_group = h5_group.create_group(
                    f'shap/{scorer.value}|{background.value}')
                _write_fields(shap_group, shap)

        elif kind == 'attr':
            h5_group.attrs[name] = value.value if isinstance(value,
                                                             Enum) else value
        elif kind == 'dataset':
            h5_group.create_dataset(name, data=np.array(value), dtype=float)


def read_sample(f: h5py.Group, sample_idx: int) -> SampleWithSHAP:
    return _read_fields(f[str(sample_idx)], SampleWithSHAP)


def _read_fields(h5_group: h5py.Group, cls):
    kwargs = {}
    for f_ in fields(cls):
        name, kind, enum = f_.name, _kind(f_), _enum(f_)
        if name == 'feature_vals':
            kwargs[name] = _read_fields(h5_group, SampleValues)
        elif name == 'shap_vals':
            results = [
                _read_fields(g, SHAPResult)
                for g in h5_group.get('shap', {}).values()
            ]
            kwargs[name] = {(r.predictor, r.background): r for r in results}
        elif kind == 'attr' and name in h5_group.attrs:
            value = _from_attr(h5_group.attrs[name])
            kwargs[name] = value if enum is None else (enum(value))
        elif kind == 'dataset' and name in h5_group:
            kwargs[name] = h5_group[name][()]
    return cls(**kwargs)


def _from_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.dtype == object:
        return [_from_attr(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dumps(obj) -> str:
    return json.dumps(to_json_safe(obj))


def write_metadata(f: h5py.Group, meta: SHAPFileMetadata) -> None:
    f.attrs['interpret'] = _dumps(meta.interpret)
    f.attrs['prediction_target'] = _dumps(meta.prediction_target)
    if meta.resolved is not None:
        f.attrs['resolved'] = _dumps(meta.resolved)
    if meta.scenario is not None:
        f.attrs['scenario'] = meta.scenario


def read_metadata(f: h5py.Group) -> SHAPFileMetadata:
    resolved = f.attrs.get('resolved')
    return SHAPFileMetadata(
        interpret=load_interpret_cfg(json.loads(f.attrs['interpret'])),
        prediction_target=load_prediction_target(
            json.loads(f.attrs['prediction_target'])),
        scenario=_from_attr(f.attrs['scenario'])
        if 'scenario' in f.attrs else None,
        resolved=None if resolved is None else ResolvedValues(
            **json.loads(resolved)),
        num_samples=len(sample_indices(f)),
    )


def sample_indices(f: h5py.Group) -> list[int]:
    return sorted(int(k) for k in f.keys() if k.isdigit())


def load_samples(path: Path) -> tuple[SHAPFileMetadata, SampleWithSHAP]:
    with h5py.File(path, 'r') as f:
        return read_metadata(f), [read_sample(f, i) for i in sample_indices(f)]
