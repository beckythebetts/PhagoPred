import json
from dataclasses import fields

from PhagoPred.utils.logger import get_logger

from .models import LSTMCfg, CNNCfg, RSFCfg, ModelCfg
from .attention import AttentionCfg
from .losses import SurvivalLossCfg, BinaryLossCfg
from .datasets import DatasetCfg, BinaryDatasetCfg, SurvivalDatasetCfg
from .training import TrainingCfg

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
    datset_type = BinaryDatasetCfg if d['num_bins'] == 1 else SurvivalDatasetCfg
    log.debug(f'Loading datclass cfg: {datset_type}')
    return _load_dataclass(datset_type, d)


def _load_loss(d: dict):
    if d['is_binary']:
        cls = BinaryLossCfg
    else:
        cls = SurvivalLossCfg
    return _load_dataclass(cls, d)


def load_experiment_cfg(path: str) -> ExperimentCfg:
    """Load a full experimnt config from .json."""
    with path.open('r') as f:
        d = json.load(f)

    return ExperimentCfg(
        model=_load_model(d['model']),
        attention=_load_dataclass(AttentionCfg, d['attention']),
        loss=_load_loss(d['loss']),
        dataset=_load_dataset(d['dataset']),
        training=_load_dataclass(TrainingCfg, d['training']),
        feature_combo=d['feature_combo'],
    )
