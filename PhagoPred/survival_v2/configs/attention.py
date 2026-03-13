from dataclasses import dataclass, field

from typing import Literal


@dataclass
class AttentionCfg:
    """Config for attention mechanism"""
    attention_type: Literal[
        'mean_pool',
        'last_pool',
        'vector',
        'fc',
        'multihead',
    ]
    hidden_dims: list = field(default_factory=list)  # for FC only
    num_heads: int = 0  # For multihead only
    dropout: float = 0  # for multihead only
    embed_dim: int = 0  #


ATTENTION = {
    'Mean Pooling': AttentionCfg('mean_pool'),
    'Last': AttentionCfg('last_pool'),
    'Vector': AttentionCfg('vector'),
    'FC Small': AttentionCfg('fc', hidden_dims=[32]),
    'FC Medium': AttentionCfg('fc', hidden_dims=[64, 32]),
    'Multihead': AttentionCfg('multihead', num_heads=2, dropout=0.1),
}
# ATTENTION = {
#     'none_mean': {
#         'type': 'mean_pool'
#     },
#     'none_last': {
#         'type': 'last_pool'
#     },
#     'vector': {
#         'type': 'vector'
#     },
#     'fc_small': {
#         'type': 'fc',
#         'hidden_dims': [32]
#     },
#     'fc_medium': {
#         'type': 'fc',
#         'hidden_dims': [64, 32]
#     },
#     'multihead_2': {
#         'type': 'multihead',
#         'num_heads': 2,
#         'dropout': 0.1
#     },
#     # 'multihead_4': {
#     #     'type': 'multihead',
#     #     'num_heads': 4,
#     #     'dropout': 0.1
#     # },
# }
