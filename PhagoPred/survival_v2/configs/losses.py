from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LossCfg:
    """Abstract class for loss function configuration"""
    is_binary: bool = field(default=False, init=False)


@dataclass
class SurvivalLossCfg(LossCfg):
    """Loss funciton configuration for survival models/datasets"""
    nll: float = 0.0
    nll_type: Literal['standard', 'soft_target'] = 'standard'
    soft_target_sigma: float = 0.8
    ranking: float = 0.0
    ranking_type: Literal['concordance', 'cif'] = 'concordance'
    prediction: float = 0.0
    name: str = ''
    bin_weights: list = field(default=None, compare=False)


@dataclass
class BinaryLossCfg(LossCfg):
    """Loss functions for binary models/datasets"""
    loss_type: Literal['bce', 'weighted_bce', 'focal']
    focal_alpha: float = 0.0
    focal_gamma: float = 0.0
    pos_weight: float = field(default=0.0, compare=False)
    is_binary: bool = field(default=True, init=False)
    name: str = ''


LOSSES = {
    'NLL':
    SurvivalLossCfg(nll=1.0, nll_type='standard', name='NLL'),
    'Soft Target NLL':
    SurvivalLossCfg(nll=1.0, nll_type='soft_target', name='Soft Target NLL'),
    'NLL + Ranking':
    SurvivalLossCfg(
        nll=1.0,
        nll_type='standard',
        ranking=1.0,
        ranking_type='concordance',
        name='NLL + Ranking',
    ),
    'Soft NLL + Ranking':
    SurvivalLossCfg(
        nll=1.0,
        nll_type='soft_target',
        ranking=1.0,
        ranking_type='concordance',
        name='Soft NLL + Ranking',
    ),
    'NLL + Ranking + Prediction':
    SurvivalLossCfg(nll=1.0,
                    ranking=1.0,
                    prediction=0.3,
                    name='NLL + Ranking + Prediction'),
    'BCE':
    BinaryLossCfg('bce', name='BCE'),
    'Weighted BCE':
    BinaryLossCfg('weighted_bce', name='Weighted BCE'),
    'Focal Loss':
    BinaryLossCfg('focal',
                  focal_alpha=0.25,
                  focal_gamma=2.0,
                  name='Focal Loss'),
}
# LOSSES = {
#     'nll_only': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.8,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target_tight': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.5,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'soft_target_wide': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 1.0,
#         'ranking': 0.0,
#         'prediction': 0.0
#     },
#     'nll_ranking_weak': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.05,
#         'ranking_type': 'concordance',
#         'prediction': 0.0
#     },
#     'nll_ranking_strong': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.2,
#         'ranking_type': 'concordance',
#         'prediction': 0.0
#     },
#     'soft_target_ranking': {
#         'nll': 1.0,
#         'nll_type': 'soft_target',
#         'soft_target_sigma': 0.8,
#         'ranking': 0.05,
#         'ranking_type': 'concordance',
#         'prediction': 0.0
#     },
#     'nll_prediction': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.0,
#         'prediction': 0.5
#     },
#     'full': {
#         'nll': 1.0,
#         'nll_type': 'standard',
#         'ranking': 0.1,
#         'ranking_type': 'concordance',
#         'prediction': 0.5
#     },
#     'bce': {
#         'bce': 1.0,
#     },
#     'weighted_bce': {
#         'weighted_bce': 1.0,
#     },
#     'binary_focal': {
#         'binary_focal': 1.0,
#         'binary_focal_alpha': 0.25,
#         'binary_focal_gamma': 2.0,
#     },
# }
