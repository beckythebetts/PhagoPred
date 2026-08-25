from __future__ import annotations
import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path, PurePath

import numpy as np

from PhagoPred.utils.logger import get_logger

from .models import LSTMCfg, CNNCfg, RSFCfg, ModelCfg
from .attention import AttentionCfg
from .losses import SurvivalLossCfg, BinaryLossCfg
from .datasets import (DATASET_TYPES, DatasetCfg, BinaryDatasetCfg,
                       SurvivalDatasetCfg)
from .training import TrainingCfg
from .calibration import CalibrationCfg, CALIBRATION_TYPES

from .experiments import ExperimentCfg

log = get_logger()

MODEL_TYPES = {'LSTM': LSTMCfg, 'CNN': CNNCfg, 'RSF': RSFCfg}
LOSS_TYPES = {'Survival': SurvivalLossCfg, 'Binary': BinaryLossCfg}


def _load_dataclass(cls, d: dict):
    """Recursively construct a dataclass from a dict, ignoring init=False fields."""
    log.debug(f'Loading dataclass object {cls}')
    init_fields = {f.name for f in fields(cls) if f.init}
    kwargs = {k: v for k, v in d.items() if k in init_fields}
    return cls(**kwargs)


def _load_model(d: dict) -> ModelCfg:
    type_name = d.pop('model_type')  # e.g. "LSTMCfg"
    log.debug(f'Loading model cfg: {type_name}')
    cls = MODEL_TYPES[type_name]
    return _load_dataclass(cls, d)


def _load_dataset(d: dict) -> DatasetCfg:
    dataset_type = d.get('dataset_type')
    if dataset_type in DATASET_TYPES:
        cls = DATASET_TYPES[dataset_type]
    else:
        # Configs saved before dataset_type existed: fall back to the old
        # num_bins heuristic. Those predate BinaryClassDatasetCfg (whose Path
        # train_paths could not be json-dumped), so binary/survival covers them.
        cls = BinaryDatasetCfg if d['num_bins'] == 1 else SurvivalDatasetCfg
        if dataset_type:
            log.warning(f'Unknown dataset_type {dataset_type!r}; '
                        f'falling back to {cls.__name__}')
    log.debug(f'Loading datclass cfg: {cls}')
    return _load_dataclass(cls, d)


def _load_loss(d: dict):
    if d['is_binary']:
        cls = BinaryLossCfg
    else:
        cls = SurvivalLossCfg
    return _load_dataclass(cls, d)


def _load_calibration(d: dict | None) -> CalibrationCfg | None:
    if d is None:
        return None
    cal_type = d.get('calibration_type', 'none')
    # if cal_type == 'none' or cal_type not in CALIBRATION_TYPES:
    #     cal_type =
    cls = CALIBRATION_TYPES[cal_type]
    return _load_dataclass(cls, d)


def load_experiment_cfg(path: str | Path) -> ExperimentCfg:
    """Load a full experimnt config from .json."""
    with Path(path).open('r') as f:
        d = json.load(f)

    return ExperimentCfg(
        model=_load_model(d['model']),
        attention=_load_dataclass(AttentionCfg, d['attention']),
        loss=_load_loss(d['loss']),
        dataset=_load_dataset(d['dataset']),
        training=_load_dataclass(TrainingCfg, d['training']),
        feature_combo=d['feature_combo'],
        calibration=_load_calibration(d.get('calibration')),
    )


def _json_key(key) -> str:
    """JSON object keys must be strings; Paths etc. are stringified."""
    if isinstance(key, PurePath):
        return str(key)
    if isinstance(key, np.generic):
        return str(key.item())
    return key if isinstance(key, str) else str(key)


def to_json_safe(obj):
    """Recursively convert an object into json-serialisable primitives.

    Handles the non-primitive types that end up on config dataclasses:
    Paths (-> str), numpy scalars/arrays and torch tensors (-> float/list),
    tuples and sets (-> list), Enums (-> value) and nested dataclasses.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {_json_key(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, PurePath):
        return str(obj)
    if isinstance(obj, Enum):
        return to_json_safe(obj.value)
    if isinstance(obj, np.generic):
        return obj.item()  # np.float32 -> float
    if isinstance(obj, np.ndarray):
        return to_json_safe(obj.tolist())
    if hasattr(obj, 'detach') and hasattr(obj, 'tolist'):  # torch.Tensor
        return to_json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    log.warning(f'No json conversion for {type(obj)}, falling back to str')
    return str(obj)


def experiment_cfg_to_dict(cfg: ExperimentCfg) -> dict:
    """Flatten an ExperimentCfg into a json-serialisable dict.

    Same shape as dataclasses.asdict(cfg), but with Paths, numpy values and
    tensors converted so json.dump() cannot fail on it.
    """
    return to_json_safe(cfg)


def save_experiment_cfg(cfg: ExperimentCfg,
                        path: str | Path,
                        indent: int = 2) -> Path:
    """Save a full experiment config to .json. Inverse of load_experiment_cfg."""
    path = Path(path)
    log.debug(f'Saving experiment cfg to {path}')
    # Serialise before opening the file: json.dump() writes incrementally, so a
    # failure part way through leaves a truncated config.json on disk.
    text = json.dumps(experiment_cfg_to_dict(cfg), indent=indent)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
