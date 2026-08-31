from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Type
from collections import defaultdict
import numpy as np
import h5py

from PhagoPred.utils.logger import get_logger

log = get_logger()


@dataclass
class DatasetCfg:
    """Config to hold datset information.

    dataset_type tags which subclass this is so a saved config.json can be
    loaded back as the same type (see DATASET_TYPES and configs.io).
    """
    dataset_type: str = field(default='', init=False)
    num_bins: int
    train_paths: list[Union[Path, str]] = field(default=list, compare=False)
    val_paths: list[Union[Path, str]] = field(default=list, compare=False)
    min_length: int = 50

    name: str = ''
    calibrate_paths: list[Union[Path, str]] = field(default=list,
                                                    compare=False)

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
    dataset_type: str = field(default='survival', init=False)
    max_time_to_death: int = 0
    name: str = ''
    bins: list[int] | None = field(default=None, compare=False)


@dataclass
class BinaryDatasetCfg(DatasetCfg):
    """Config to hold binary dataset information"""
    dataset_type: str = field(default='binary', init=False)
    prediction_horizon: int = 0
    num_bins: int = field(default=1, init=False)
    name: str = ''


@dataclass
class BinaryClassDatasetCfg(DatasetCfg):
    dataset_type: str = field(default='binary_class', init=False)
    num_bins: int = field(default=1, init=False)
    class_dict: dict = field(default_factory=dict)


DATASET_TYPES: dict[str, type[DatasetCfg]] = {
    'survival': SurvivalDatasetCfg,
    'binary': BinaryDatasetCfg,
    'binary_class': BinaryClassDatasetCfg,
}


def generate_kfold_dataset_configs(all_paths: list[Path], num_folds: int,
                                   cfg_type: type[DatasetCfg], name: str,
                                   **kwargs) -> list[DatasetCfg]:
    log.info('Generating kfold datset config')
    cfgs = []
    all_paths = list(all_paths)
    np.random.shuffle(all_paths)

    class_dict = kwargs.get('class_dict')

    if class_dict is None:
        split = np.array_split(all_paths, num_folds)
        for k in range(num_folds):
            val_paths = split[k].tolist()
            train_paths = np.concatenate(
                [a for i, a in enumerate(split) if i != k]).tolist()
            cfgs.append(
                cfg_type(
                    train_paths=train_paths,
                    val_paths=val_paths,
                    calibrate_paths=val_paths,
                    name=f'{name}',
                    **kwargs,
                ))
    else:
        # Stratified smapling, ensuring always even number of datsets per class, if class_dict exists
        label_to_paths = defaultdict(list)
        for path, label in class_dict.items():
            label_to_paths[label].append(path)

        splits = {}
        for label, paths in label_to_paths.items():
            paths = list(paths)
            np.random.shuffle(paths)
            splits[label] = np.array_split(paths, num_folds)

        for k in range(num_folds):
            val_paths = np.concatenate([split[k] for split in splits.values()
                                        ]).tolist()
            train_paths = np.concatenate([
                a for split in splits.values() for i, a in enumerate(split)
                if i != k
            ]).tolist()
            np.random.shuffle(train_paths)
            cfgs.append(
                cfg_type(
                    train_paths=train_paths,
                    val_paths=val_paths,
                    calibrate_paths=val_paths,
                    name=f'{name}',
                    **kwargs,
                ))

    return cfgs


def generate_fluor_class_dict(paths: list[Path]) -> dict[Path, str]:
    log.info('Generating class dict for dataset')
    fluor = 1
    no_fluor = 0
    class_dict = {}
    for path in paths:
        with h5py.File(path, 'r') as f:
            if 'Epi' in f['Images']:
                class_dict[path] = fluor
            else:
                class_dict[path] = no_fluor
    return class_dict


def get_kfold_dataset() -> list:
    paths = list(
        Path('~/thor_server/MacrophageData/24_07/').expanduser().glob('*.h5'))
    return generate_kfold_dataset_configs(
        all_paths=paths,
        num_folds=4,
        cfg_type=BinaryClassDatasetCfg,
        name='24_07',
        min_length=50,
        class_dict=generate_fluor_class_dict(paths),
    )


DATASETS = {
    '24_07_test':
    BinaryClassDatasetCfg(
        train_paths=[
            Path('~/thor_server/MacrophageData/24_07/').expanduser() /
            f'{_}.h5' for _ in ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R')
        ],
        val_paths=[
            Path('~/thor_server/MacrophageData/24_07/').expanduser() /
            f'{_}.h5' for _ in ('S', 'T', 'U', 'V', 'W', 'X')
        ],
        calibrate_paths=[
            Path('~/thor_server/MacrophageData/24_07/').expanduser() /
            f'{_}.h5' for _ in ('S', 'T', 'U', 'V', 'W', 'X')
        ],
        min_length=50,
        name='24_07_test',
        class_dict={
            p: 0
            for p in [
                Path('~/thor_server/MacrophageData/24_07/').expanduser() /
                f'{_}.h5' for _ in ('A', 'B', 'C', 'J', 'K', 'L', 'M', 'N',
                                    'O', 'V', 'W', 'X')
            ]
        } | {
            p: 1
            for p in [
                Path('~/thor_server/MacrophageData/24_07/').expanduser() /
                f'{_}.h5' for _ in ('D', 'E', 'F', 'G', 'H', 'I', 'P', 'Q',
                                    'R', 'S', 'T', 'U')
            ]
        },
    ),
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
        'Area',
        'Circularity',
        'Displacement',
        'Perimeter',
        'Phagocytes within 100 pixels',
        'Phagocytes within 250 pixels',
        'Phagocytes within 500 pixels',
        'Skeleton Branch Length Mean',
        'Skeleton Branch Length Max',
        'Skeleton Branch Length Std',
        'Skeleton Branch Points',
        'Skeleton End Points',
        'Skeleton Length',
        'Speed',
    ],
    'No Frame Count': ['0', '1', '3'],
}
