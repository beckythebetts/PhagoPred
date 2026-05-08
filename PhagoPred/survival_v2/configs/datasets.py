from dataclasses import dataclass, field
from pathlib import Path
from typing import Union


@dataclass
class DatasetCfg:
    """Config to hold datset information."""
    train_paths: list[Union[Path, str]]
    val_paths: list[Union[Path, str]]
    min_length: int
    num_bins: int
    name: str = ''

    normalisation_means: list[float] = field(default=None, compare=False)
    normalisation_stds: list[float] = field(default=None, compare=False)


@dataclass
class SurvivalDatasetCfg(DatasetCfg):
    """Config to hold survival dataset information"""
    max_time_to_death: int = 0
    name: str = ''


@dataclass
class BinaryDatasetCfg(DatasetCfg):
    """Config to hold binary dataset information"""
    prediction_horizon: int = 0
    num_bins: int = field(default=1, init=False)
    name: str = ''


DATASETS = {
    'Survival Baseline':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/baseline_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        min_length=50,
        max_time_to_death=1000,
        num_bins=5,
        name='Baseline',
    ),
    'Binary Baseline':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/baseline_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        min_length=50,
        prediction_horizon=200,
        name='Baseline',
    ),
    'Survival Challenging':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/challenging_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/challenging_val.h5'],
        min_length=50,
        max_time_to_death=1000,
        num_bins=5,
        name='Challenging',
    ),
    'Binary Challenging':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/challenging_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/challenging_val.h5'],
        min_length=50,
        prediction_horizon=200,
        name='Challenging',
    ),
    'Graph Linear':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/linear_val.h5'],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear',
    ),
    'Binary Graph Linear':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/linear_val.h5'],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear',
    ),
    'Graph Chain':
    SurvivalDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/chain_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/chain_val.h5'],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Chain',
    ),
    'Binary Graph Chain':
    BinaryDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/chain_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/chain_val.h5'],
        min_length=50,
        prediction_horizon=25,
        name='Graph Chain',
    ),
    'Graph Multiplicative':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/multiplicative_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/multiplicative_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Multiplicative',
    ),
    'Binary Graph Multiplicative':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/multiplicative_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/multiplicative_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Multiplicative',
    ),
    'Graph Resetting Accumulation':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/resetting_accumulation_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/resetting_accumulation_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Resetting Accumulation',
    ),
    'Binary Graph Resetting Accumulation':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/resetting_accumulation_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/resetting_accumulation_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Resetting Accumulation',
    ),
    'Graph Ratio':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/ratio_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/ratio_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Ratio',
    ),
    'Binary Graph Ratio':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/ratio_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/ratio_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Ratio',
    ),
    # Noise-level comparison (linear rules)
    'Graph Linear No Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_no_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_no_noise_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear No Noise',
    ),
    'Binary Graph Linear No Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_no_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_no_noise_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear No Noise',
    ),
    'Graph Linear Low Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_noise_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear Low Noise',
    ),
    'Binary Graph Linear Low Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_noise_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Low Noise',
    ),
    'Graph Linear High Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_noise_val.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear High Noise',
    ),
    'Binary Graph Linear High Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_noise_val.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear High Noise',
    ),
}
FEATURE_COMBOS = {
    'All': [
        'frame_count',
        'linear_trend',
        'polynomial_trend',
        'random_walk',
        'oscillation',
        'oscillation + linear',
    ],
    'No Frame Count': ['0', '1', '3'],
}
