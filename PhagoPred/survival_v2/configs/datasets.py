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
    class_names: list[str] = field(
        default_factory=lambda: ['No Event', 'Event'])
    #


@dataclass
class BinaryClassDatasetCfg(DatasetCfg):
    dataset_type: str = field(default='binary_class', init=False)
    num_bins: int = field(default=1, init=False)
    class_dict: dict = field(default_factory=dict)
    class_names: list[str] = field(
        default_factory=lambda: ['No Fluor', 'Fluor'])


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

        if len(splits) > 0:
            for k in range(num_folds):
                val_paths = np.concatenate(
                    [split[k] for split in splits.values()]).tolist()
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
        else:
            log.warning(
                f'No datasets found fro kfold creation from {all_paths}')

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
    try:
        return generate_kfold_dataset_configs(
            all_paths=paths,
            num_folds=4,
            cfg_type=BinaryClassDatasetCfg,
            name='24_07',
            min_length=50,
            class_dict=generate_fluor_class_dict(paths),
        )
    except ValueError:
        return []


def _graph_binary(scenario: str, name: str) -> BinaryDatasetCfg:
    """Binary config for a graph-synthetic scenario, whose train/val/cal
    files are named after the scenario's ``filename``."""
    root = 'PhagoPred/Datasets/graph_synthetic'
    return BinaryDatasetCfg(
        train_paths=[f'{root}/{scenario}_train.h5'],
        val_paths=[f'{root}/{scenario}_val.h5'],
        calibrate_paths=[f'{root}/{scenario}_cal.h5'],
        min_length=50,
        prediction_horizon=25,
        name=name,
    )


DATASETS = {
    '24_07_0':
    generate_kfold_dataset_configs(
        all_paths=list(
            Path('~/thor_server/MacrophageData/24_07/split_0').expanduser().
            glob('*.h5')),
        num_folds=4,
        cfg_type=BinaryClassDatasetCfg,
        name='Day 1',
        min_length=50,
        class_dict=generate_fluor_class_dict(
            list(
                Path('~/thor_server/MacrophageData/24_07/split_0').expanduser(
                ).glob('*.h5')))),
    '24_07_1':
    generate_kfold_dataset_configs(
        all_paths=list(
            Path('~/thor_server/MacrophageData/24_07/split_1').expanduser().
            glob('*.h5')),
        num_folds=4,
        cfg_type=BinaryClassDatasetCfg,
        name='Day 2',
        min_length=50,
        class_dict=generate_fluor_class_dict(
            list(
                Path('~/thor_server/MacrophageData/24_07/split_1').expanduser(
                ).glob('*.h5')))),
    '24_07_2':
    generate_kfold_dataset_configs(
        all_paths=list(
            Path('~/thor_server/MacrophageData/24_07/split_2').expanduser().
            glob('*.h5')),
        num_folds=4,
        cfg_type=BinaryClassDatasetCfg,
        name='Day 3',
        min_length=50,
        class_dict=generate_fluor_class_dict(
            list(
                Path('~/thor_server/MacrophageData/24_07/split_2').expanduser(
                ).glob('*.h5')))),
    '24_07':
    generate_kfold_dataset_configs(
        all_paths=list(
            Path('~/thor_server/MacrophageData/24_07').expanduser().glob(
                '*.h5')),
        num_folds=4,
        cfg_type=BinaryClassDatasetCfg,
        name='Day 3',
        min_length=50,
        class_dict=generate_fluor_class_dict(
            list(
                Path('~/thor_server/MacrophageData/24_07').expanduser().glob(
                    '*.h5')))),
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
    # Graph-synthetic scenarios, generated by
    # data/graph_synthetic/scenarios.py (ALL_CFGS). Binary configs only.
    'Binary Graph Linear':
    _graph_binary('linear', 'Graph Linear'),
    'Binary Graph Linear Chain':
    _graph_binary('linear_chain', 'Graph Linear Chain'),
    'Binary Graph Nonlinear Chain':
    _graph_binary('nonlinear_chain', 'Graph Nonlinear Chain'),
    'Binary Graph Multiplicative Hazard':
    _graph_binary('multiplicative_hazard', 'Graph Multiplicative Hazard'),
    'Binary Graph Threshold Hazard':
    _graph_binary('threshold_hazard', 'Graph Threshold Hazard'),
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
        'Skeleton Branches',
        'Skeleton Length',
        'Speed',
        'Major Axis Length',
        'Minor Axis Length',
        'Eccentricity',
    ],
    'No Frame Count': ['0', '1', '3'],
}
