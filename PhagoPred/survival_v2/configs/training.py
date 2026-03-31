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
    TrainingCfg(num_epochs=50,
                batch_size=256,
                lr=1e-3,
                scheduler='step',
                step_size=10,
                gamma=0.9,
                name='Standard'),
    'Long':
    TrainingCfg(num_epochs=300,
                batch_size=256,
                lr=1e-3,
                scheduler='step',
                step_size=10,
                gamma=0.9,
                name='Long'),
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
