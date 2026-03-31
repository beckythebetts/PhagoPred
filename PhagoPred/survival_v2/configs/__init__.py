"""
Experiment configuration templates.
"""
from .models import MODELS, ModelCfg
from .attention import ATTENTION
from .losses import LOSSES, LossCfg
from .datasets import DATASETS, FEATURE_COMBOS
from .training import TRAINING, TrainingCfg
from .experiments import generate_experiment_grid, collapse_experiment_grid, EXPERIMENT_SUITES, ExperimentCfg

__all__ = [
    'MODELS',
    'ATTENTION',
    'LOSSES',
    'DATASETS',
    'FEATURE_COMBOS',
    'TRAINING',
    'EXPERIMENT_SUITES',
    'generate_experiment_grid',
    'collapse_experiment_grid',
    'ModelCfg',
    'TrainingCfg',
    'LossCfg',
    'ExperimentCfg',
]
