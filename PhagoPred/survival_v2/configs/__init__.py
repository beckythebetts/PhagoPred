"""
Experiment configuration templates.
"""
from .experiment_config import (
    MODELS,
    ATTENTION,
    LOSSES,
    DATASETS,
    TRAINING,
    EXPERIMENT_SUITES,
    generate_experiment_grid
)

__all__ = [
    'MODELS',
    'ATTENTION',
    'LOSSES',
    'DATASETS',
    'TRAINING',
    'EXPERIMENT_SUITES',
    'generate_experiment_grid'
]
