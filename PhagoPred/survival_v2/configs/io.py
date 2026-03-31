import json
from dataclasses import fields

from PhagoPred.utils.logger import get_logger

from .models import LSTMCfg, CNNCfg, RSFCfg
from .attention import AttentionCfg
from .losses import SurvivalLossCfg, BinaryLossCfg
from .datasets import DatasetCfg
from .training import TrainingCfg

from .experiments import ExperimentCfg

log = get_logger()

MODEL_TYPES = {'LSTM': LSTMCfg, 'CNN': CNNCfg, 'RSF': RSFCfg}
LOSS_TYPES = {'Survival': SurvivalLossCfg, 'Binary': BinaryLossCfg}


def _load_dataclass(cls, d: dict):
    """Recursively construct a dataclass from a dict, ignoring init=False fields."""
    log.info(f'Loading dataclass object {cls}')
    init_fields = {f.name for f in fields(cls) if f.init}
    kwargs = {k: v for k, v in d.items() if k in init_fields}
    return cls(**kwargs)


def _load_model(d: dict):
    type_name = d.pop('model_type')  # e.g. "LSTMCfg"
    cls = MODEL_TYPES[type_name]
    return _load_dataclass(cls, d)


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
        dataset=_load_dataclass(DatasetCfg, d['dataset']),
        training=_load_dataclass(TrainingCfg, d['training']),
        feature_combo=d['feature_combo'],
    )
