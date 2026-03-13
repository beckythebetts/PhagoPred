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

    normalisation_means: list[float] = None
    normalisation_stds: list[float] = None


@dataclass
class SurvivalDatasetCfg(DatasetCfg):
    """Config to hold survival dataset information"""
    max_time_to_death: int = 0


@dataclass
class BinaryDatasetCfg(DatasetCfg):
    """Config to hold binary dataset information"""
    prediction_horizon: int = 0
    num_bins: int = field(default=1, init=False)


DATASETS = {
    'Survival Baseline':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/baseline_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        min_length=50,
        max_time_to_death=200,
        num_bins=5,
    ),
    'Binary Baseline':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/synthetic_variants/baseline_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        min_length=50,
        prediction_horizon=200,
    )
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

# DATASETS = {
#     # Original synthetic dataset
#     'synthetic_basic': {
#         'train_paths': ['PhagoPred/Datasets/synthetic.h5'],
#         'val_paths': ['PhagoPred/Datasets/val_synthetic.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins': 5,
#         'min_length': 500,
#         'max_time_to_death': 100
#     },

#     # Baseline - clean dataset with all rules
#     'baseline': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
#         'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
#         'num_bins': 5,
#         'min_length': 50,
#         'max_time_to_death': 200,
#     },
#     'baseline_binary': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
#         'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
#         'prediction_horizon': 100,
#     },
#     'baseline_large': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_large_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_large_val.h5'],
#         'num_bins':
#         5,
#         'min_length':
#         50,
#         'max_time_to_death':
#         200,
#     },

#     # Single rule variants
#     'single_rule_variation': {
#         'train_paths': [
#             'PhagoPred/Datasets/synthetic_variants/single_rule_variation_train.h5'
#         ],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/single_rule_variation_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },
#     'single_rule_ramp': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/single_rule_ramp_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/single_rule_ramp_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },
#     'no_rules': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/no_rules_train.h5'],
#         'val_paths': ['PhagoPred/Datasets/synthetic_variants/no_rules_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins': 5,
#         'min_length': 500,
#         'max_time_to_death': 100
#     },

#     # Different noise levels
#     'low_noise': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/low_noise_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/low_noise_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins': 5,
#         'min_length': 500,
#         'max_time_to_death': 100
#     },
#     'medium_noise': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/medium_noise_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/medium_noise_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },
#     'high_noise': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/high_noise_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/high_noise_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },

#     # Late entry variants
#     'late_entry_30pct': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_30pct_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_30pct_val.h5'],
#         # 'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         50,
#         'max_time_to_death':
#         100
#     },
#     'late_entry_50pct': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_50pct_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_50pct_val.h5'],
#         # 'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         50,
#         'max_time_to_death':
#         100
#     },
#     'late_entry_extreme': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_val.h5'],
#         # 'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         50,
#         'max_time_to_death':
#         200,
#     },
#     'late_entry_extreme_start_feature': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_val.h5'],
#         # 'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         50,
#         'max_time_to_death':
#         100,
#         'add_start_frame_feature':
#         True,
#     },

#     # Missing data variants
#     'missing_data_10pct': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/missing_data_10pct_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/missing_data_10pct_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },
#     'missing_data_20pct': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/missing_data_20pct_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/missing_data_20pct_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },

#     # Combined challenges
#     'challenging': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/challenging_train.h5'],
#         'val_paths':
#         ['PhagoPred/Datasets/synthetic_variants/challenging_val.h5'],
#         'features': ['0', '1', '2', '3'],
#         'num_bins':
#         5,
#         'min_length':
#         500,
#         'max_time_to_death':
#         100
#     },

#     # Feature subset experiments (using baseline dataset)
#     'features_012': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
#         'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
#         'features': ['0', '1', '2'],
#         'num_bins': 5,
#         'min_length': 500,
#         'max_time_to_death': 100
#     },
#     'features_01': {
#         'train_paths':
#         ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
#         'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
#         'features': ['0', '1'],
#         'num_bins': 5,
#         'min_length': 500,
#         'max_time_to_death': 100
#     },
# }
