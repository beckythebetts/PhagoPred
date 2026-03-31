from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelCfg:
    """Abstract base class for model configs"""
    is_classical: bool = field(default=False, init=False)
    model_type: Literal['CNN', 'LSTM', 'RSF'] = field(default='', init=False)


@dataclass
class LSTMCfg(ModelCfg):
    """Config for LSTM model"""
    hidden_size: int
    num_layers: int
    dropout: float
    fc_layers: list
    predictor_layers: list
    model_type: Literal['CNN', 'LSTM', 'RSF'] = field(default='LSTM',
                                                      init=False)
    name: str = ''


@dataclass
class CNNCfg(ModelCfg):
    """Config ofor CNN model"""
    num_channels: list
    kernel_sizes: list
    dilations: list
    fc_layers: list
    model_type: Literal['CNN', 'LSTM', 'RSF'] = field(default='CNN',
                                                      init=False)
    name: str = ''


@dataclass
class RSFCfg(ModelCfg):
    """Config for random forest/random surival forest model"""
    n_estimators: int
    max_depth: int
    window_sizes: list
    is_classical: bool = field(default=True, init=False)
    model_type: Literal['CNN', 'LSTM', 'RSF'] = field(default='RSF',
                                                      init=False)
    name: str = ''


MODELS = {
    'LSTM Medium':
    LSTMCfg(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        fc_layers=[32],
        predictor_layers=[],
        name='LSTM Medium',
    ),
    'CNN Medium':
    CNNCfg(num_channels=[64] * 9,
           kernel_sizes=[3] * 9,
           dilations=[2**n for n in range(9)],
           fc_layers=[64, 32],
           name='CNN Medium'),
    'Random Forest':
    RSFCfg(
        n_estimators=100,
        max_depth=3,
        window_sizes=[25, 100, 500, 1000],
        name='Random Forest',
    ),
}

# MODELS = {
#     'lstm_small': {
#         'type': 'lstm',
#         'lstm_hidden_size': 32,
#         'lstm_num_layers': 1,
#         'lstm_dropout': 0.0,
#         'fc_layers': [32],
#         'use_predictor': False
#     },
#     'lstm_medium': {
#         'type': 'lstm',
#         'lstm_hidden_size': 64,
#         'lstm_num_layers': 2,
#         'lstm_dropout': 0.2,
#         'fc_layers': [64, 32],
#         'use_predictor': False
#     },
#     'lstm_large': {
#         'type': 'lstm',
#         'lstm_hidden_size': 128,
#         'lstm_num_layers': 2,
#         'lstm_dropout': 0.2,
#         'predictor_layers': [64],
#         'fc_layers': [128, 64],
#         'use_predictor': True
#     },
#     'cnn_small': {
#         'type': 'cnn',
#         'num_channels': [32, 32, 32],
#         'kernel_sizes': [3, 3, 3],
#         'dilations': [1, 2, 4],
#         'fc_layers': [32]
#     },
#     'cnn_medium': {
#         'type': 'cnn',
#         'num_channels': [64] * 9,
#         'kernel_sizes': [3] * 9,
#         'dilations': [2**n for n in range(9)
#                       ],  # 9 dilations covers 1000 frames with kernel size 3
#         'fc_layers': [64, 32]
#     },
#     'cnn_large': {
#         'type': 'cnn',
#         'num_channels': [128] * 10,
#         'kernel_sizes': [3] * 10,
#         'dilations': [2**n for n in range(10)],
#         'fc_layers': [128, 64, 32]
#     },

#     # ========== Classical ML Models (scikit-survival) ==========

#     # Random Survival Forest
#     'rsf': {
#         'type': 'random_survival_forest',
#         'n_estimators': 100,
#         'max_depth': None,
#         'window_sizes': [25, 100, 500, 1000],
#     },

#     # Gradient Boosting Survival Analysis
#     'gbsa': {
#         'type': 'gradient_boosting_survival',
#         'n_estimators': 100,
#         'max_depth': 3,
#         'learning_rate': 0.1,
#         'window_sizes': [100, 500, 1000],
#     },

#     # Cox Proportional Hazards
#     'coxph': {
#         'type': 'coxph',
#         'alpha': 0.1,  # L2 regularization to prevent ill-conditioning
#         'ties': 'breslow',
#         'window_sizes': [100, 500, 1000],
#     },
#     'coxph_regularized': {
#         'type': 'coxph',
#         'alpha': 0.1,  # L2 regularization
#         'ties': 'breslow',
#         'window_sizes': [100, 500, 1000],
#     },
# }
