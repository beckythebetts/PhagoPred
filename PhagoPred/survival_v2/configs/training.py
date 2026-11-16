from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainingCfg:
    """Dataclass to hold training configuration"""
    num_epochs: int
    batch_size: int
    lr: float
    step_size: int
    gamma: float
    scheduler: Literal[None, 'step'] = None
    # Dense supervision: one sample per cell, loss at every valid landmark in
    # a single forward pass (binary datasets, causal models with last_pool
    # only). batch_size then counts cells, not landmarks.
    dense: bool = False
    # Early stopping: stop after this many epochs without a new best
    # validation loss (None trains for all num_epochs). The best-validation
    # weights are kept either way.
    patience: int | None = None
    name: str = ''


TRAINING = {
    'Quick':
    TrainingCfg(num_epochs=5,
                batch_size=256,
                lr=1e-3,
                scheduler='step',
                step_size=10,
                gamma=0.9,
                name='Quick'),
    'Standard':
    TrainingCfg(num_epochs=150,
                batch_size=2048,
                lr=1e-3,
                scheduler='step',
                step_size=10,
                gamma=0.99,
                name='Standard'),
    'Long':
    TrainingCfg(num_epochs=300,
                batch_size=256,
                lr=1e-3,
                scheduler='step',
                step_size=10,
                gamma=0.9,
                name='Long'),
    'Dense':
    TrainingCfg(num_epochs=100,
                batch_size=256,
                lr=1e-3,
                scheduler='step',
                step_size=15,
                gamma=0.5,
                dense=True,
                patience=15,
                name='Dense'),
}

# TRAINING = {
#     'quick_test': {
#         'num_epochs': 10,
#         'batch_size': 128,
#         'lr': 1e-3,
#         'scheduler': 'step',
#         'step_size': 5,
#         'gamma': 0.9
#     },
#     'standard': {
#         'num_epochs': 150,
#         'batch_size': 256,
#         'lr': 1e-3,
#         'scheduler': 'step',
#         'step_size': 10,
#         'gamma': 0.9
#     },
#     'long': {
#         'num_epochs': 250,
#         'batch_size': 256,
#         'lr': 1e-3,
#         'scheduler': 'step',
#         'step_size': 15,
#         'gamma': 0.9
#     },
# }
