from __future__ import annotations
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
    calibrate_paths: list[Union[Path, str]] | None = None
    num_cells: int | None = None

    normalisation_means: list[float] = field(default=None,
                                             compare=False,
                                             repr=False)
    normalisation_stds: list[float] = field(default=None,
                                            compare=False,
                                            repr=False)


@dataclass
class SurvivalDatasetCfg(DatasetCfg):
    """Config to hold survival dataset information"""
    max_time_to_death: int = 0
    name: str = ''
    bins: list[int] | None = field(default=None, compare=False)

    # total_hazard_var: list[float] | None = None
    # unobserved_hazard_var: list[float] | None = None
    # total_pmf_var: list[float] | None = None
    # unobserved_pmf_var: list[float] | None = None


@dataclass
class BinaryDatasetCfg(DatasetCfg):
    """Config to hold binary dataset information"""
    prediction_horizon: int = 0
    num_bins: int = field(default=1, init=False)
    name: str = ''

    # total_cdf_var: float | None = None
    # unobserved_cdf_var: float | None = None


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
            'PhagoPred/Datasets/graph_synthetic/base_linear_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_linear_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear',
    ),
    'Binary Graph Linear':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_linear_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear',
    ),
    'Graph Chain':
    SurvivalDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/base_chain_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_chain_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_chain_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Chain',
    ),
    'Binary Graph Chain':
    BinaryDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/base_chain_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_chain_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_chain_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Chain',
    ),
    'Graph Multiplicative':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Multiplicative',
    ),
    'Binary Graph Multiplicative':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_multiplicative_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Multiplicative',
    ),
    'Graph Resetting Accumulation':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_resetting_accumulation_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_resetting_accumulation_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_resetting_accumulation_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Resetting Accumulation',
    ),
    'Graph Ratio':
    SurvivalDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/base_ratio_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_ratio_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_ratio_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Ratio',
    ),
    'Binary Graph Ratio':
    BinaryDatasetCfg(
        train_paths=['PhagoPred/Datasets/graph_synthetic/base_ratio_train.h5'],
        val_paths=['PhagoPred/Datasets/graph_synthetic/base_ratio_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_ratio_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Ratio',
    ),
    # Autocorrelation-only: hidden Hazard is an AR leaky-integrator of a
    # white-noise driver A (no lags). Memory lives purely in the autocorrelation.
    'Graph Autocorr':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Autocorr',
    ),
    'Binary Graph Autocorr':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_autocorr_hazard_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Autocorr',
    ),
    # Noise-level comparison (linear rules)
    'Graph Linear No Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear No Noise',
    ),
    'Binary Graph Linear No Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_no_noise_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear No Noise',
    ),
    'Graph Linear Low Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear Low Noise',
    ),
    'Binary Graph Linear Low Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_low_noise_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Low Noise',
    ),
    'Graph Linear High Noise':
    SurvivalDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_cal.h5'
        ],
        min_length=50,
        max_time_to_death=50,
        num_bins=5,
        name='Graph Linear High Noise',
    ),
    'Binary Graph Linear High Noise':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/base_linear_high_noise_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear High Noise',
    ),
    # AR-variant scenarios generated by
    # data/graph_synthetic/scenarios.py (_SCENARIO_BUILDERS x _AR_VARIANTS).
    # Binary configs only for now.
    'Binary Graph Linear Low AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_ar_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/linear_low_ar_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_low_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Low AR',
    ),
    'Binary Graph Linear Chain Low AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_low_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_low_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_low_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Chain Low AR',
    ),
    'Binary Graph Nonlinear Chain Low AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_low_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_low_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_low_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Nonlinear Chain Low AR',
    ),
    'Binary Graph Linear Med AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_med_ar_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/linear_med_ar_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_med_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Med AR',
    ),
    'Binary Graph Linear Chain Med AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_med_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_med_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_med_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Chain Med AR',
    ),
    'Binary Graph Nonlinear Chain Med AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_med_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_med_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_med_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Nonlinear Chain Med AR',
    ),
    'Binary Graph Linear High AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_ar_train.h5'
        ],
        val_paths=['PhagoPred/Datasets/graph_synthetic/linear_high_ar_val.h5'],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_high_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear High AR',
    ),
    'Binary Graph Linear Chain High AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_high_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_high_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/linear_chain_high_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Linear Chain High AR',
    ),
    'Binary Graph Nonlinear Chain High AR':
    BinaryDatasetCfg(
        train_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_high_ar_train.h5'
        ],
        val_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_high_ar_val.h5'
        ],
        calibrate_paths=[
            'PhagoPred/Datasets/graph_synthetic/nonlinear_chain_high_ar_cal.h5'
        ],
        min_length=50,
        prediction_horizon=25,
        name='Graph Nonlinear Chain High AR',
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
