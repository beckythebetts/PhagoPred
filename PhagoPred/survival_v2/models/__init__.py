"""
Survival models with pluggable attention mechanisms.
Includes both deep learning and classical ML models.
"""
from .lstm_survival import LSTMSurvival
from .cnn_survival import CNNSurvival
from .base import SurvivalModel

# Classical ML models (scikit-survival based)
from .classical_base import ClassicalSurvivalModel
from .survival_forest import RandomSurvivalForestModel, GradientBoostingSurvivalModel, CoxPHModel

__all__ = [
    # Deep learning models
    'LSTMSurvival',
    'CNNSurvival',
    'SurvivalModel',
    # Classical ML models (scikit-survival)
    'ClassicalSurvivalModel',
    'RandomSurvivalForestModel',
    'GradientBoostingSurvivalModel',
    'CoxPHModel',
]
