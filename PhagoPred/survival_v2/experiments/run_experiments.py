from pathlib import Path
import json
from datetime import datetime
from dataclasses import asdict

import torch
import numpy as np
from torch.utils.data import DataLoader

# from PhagoPred.survival_v2.configs.experiment_config import (MODELS, ATTENTION,
#                                                              LOSSES, DATASETS,
#                                                              TRAINING,
#                                                              EXPERIMENT_SUITES,
#                                                              FEATURE_COMBOS)
from PhagoPred.survival_v2.configs.experiments import EXPERIMENT_SUITES, ExperimentCfg, ModelCfg, AttentionCfg, DatasetCfg
from PhagoPred.survival_v2.configs.models import LSTMCfg, CNNCfg, RSFCfg
from PhagoPred.survival_v2.configs.datasets import SurvivalDatasetCfg, BinaryDatasetCfg
from PhagoPred.survival_v2.models.build import build_model
from PhagoPred.survival_v2.train.train import train
from PhagoPred.survival_v2.evaluate import evaluate
from PhagoPred.survival_v2.data import (
    CellDataset,
    SurvivalCellDataset,
    BinaryCellDataset,
    binary_collate_fn,
    survival_collate_fn,
)
from PhagoPred.survival_v2.utils.plots import plot_losses
from PhagoPred.survival_v2.interpret.run_interpret import interpret
from PhagoPred.utils.logger import get_logger
from PhagoPred.utils.tools import highlight_str
from .plots.plot_experiments import (
    plot_experiment_results,
    # plot_confusion_matrices,
    # plot_experiment_losses,
)

log = get_logger()


def build_datasets(
        dataset_config: DatasetCfg,
        feature_combo: list,
        is_classical: bool = False) -> tuple[CellDataset, CellDataset]:
    """
    Build train and validation datasets.

    Args:
        dataset_config: dict with dataset parameters
        feature_combo: list of feature names to include
        is_classical: If false, normalise dataset and don't use fixed length

    Returns:
        train_dataset: CellDataset for training
        val_dataset: CellDataset for validation
    """
    model_type = 'classical' if is_classical else 'deep'

    if isinstance(dataset_config, BinaryDatasetCfg):

        dataset_args = {
            # 'hdf5_paths': dataset_config['train_paths'],
            'feature_names': feature_combo,
            'min_length': dataset_config.min_length,
            'model_type': model_type,
            'prediction_horizon': dataset_config.prediction_horizon,
        }
        train_dataset = BinaryCellDataset(
            hdf5_paths=dataset_config.train_paths, **dataset_args)

        means, stds = train_dataset.get_normalisation_stats()

        val_dataset = BinaryCellDataset(hdf5_paths=dataset_config.val_paths,
                                        means=means,
                                        stds=stds,
                                        **dataset_args)

    else:
        dataset_args = {
            # 'hdf5_paths': dataset_config['train_paths'],
            'feature_names': feature_combo,
            'min_length': dataset_config.min_length,
            'model_type': model_type,
            'num_bins': dataset_config.num_bins,
            'max_time_to_death': dataset_config.max_time_to_death,
        }

        train_dataset = SurvivalCellDataset(
            hdf5_paths=dataset_config.train_paths, **dataset_args)

        event_time_bins = train_dataset.get_bins()
        means, stds = train_dataset.get_normalisation_stats()

        val_dataset = SurvivalCellDataset(
            hdf5_paths=dataset_config.val_paths,
            event_time_bins=event_time_bins,
            means=means,
            stds=stds,
            **dataset_args,
        )

    if not is_classical:
        train_dataset.apply_normalisation()
        val_dataset.apply_normalisation()

    setattr(dataset_config, 'normalisation_means', means.tolist())
    setattr(dataset_config, 'normalisation_stds', stds.tolist())
    return train_dataset, val_dataset


def run_single_experiment(
    experiment_config: ExperimentCfg,
    output_dir: Path,
    device='cuda' if torch.cuda.is_available() else 'cpu',
) -> tuple[dict, Path]:
    """
    Run a single experiment.

    Args:
        experiment_config: dict with experiment configuration
        output_dir: path to save results
        device: torch device

    Returns:
        results: dict with experiment results
    """
    log.info(f'Running Experiment {asdict(experiment_config)}')

    file_idx = 0
    exp_dir = Path(output_dir) / f"experiment_{file_idx:02d}"
    while exp_dir.exists():
        file_idx += 1
        exp_dir = Path(output_dir) / f"experiment_{file_idx:02d}"
    exp_dir.mkdir(parents=True)

    print(f"Building datasets {experiment_config.dataset.name}...")
    log.info(
        f'Building train/validatin datasets: \n\t{asdict(experiment_config.dataset)}'
    )
    train_dataset, val_dataset = build_datasets(
        experiment_config.dataset,
        experiment_config.feature_combo,
        is_classical=experiment_config.model.is_classical)

    if isinstance(train_dataset, BinaryCellDataset):

        pos_weight = train_dataset.get_pos_weight()
        setattr(experiment_config.loss, 'pos_weight', pos_weight)
        log.info(f"Computed pos_weight: {pos_weight:.2f}")

    else:
        train_dataset: SurvivalCellDataset

        bins = train_dataset.event_time_bins
        log.info(f"Event time bins: {bins}")

        bin_weights = train_dataset.get_bin_weights()
        setattr(experiment_config.loss, 'bin_weights', bin_weights)
        log.info(f"Computed bin weights: {bin_weights}")

    print(f"Building model {experiment_config.model.name}...")
    num_features = len(experiment_config.feature_combo)
    num_bins = experiment_config.dataset.num_bins
    log.info(
        f'Building model :\n\t{asdict(experiment_config.model)}\n\t{asdict(experiment_config.attention)}\n\tnum_bins = {num_bins}\n\tnum_features={num_features}'
    )
    model = build_model(experiment_config.model,
                        experiment_config.attention,
                        num_features,
                        num_bins,
                        device,
                        feature_names=experiment_config.feature_combo)

    with (exp_dir / 'config.json').open('w') as f:
        json.dump(asdict(experiment_config), f, indent=2)

    print('Training Model...')
    log.info(
        f'Training Model:\n\t{asdict(experiment_config.training)}\n\t{asdict(experiment_config.loss)}'
    )
    history = train(model,
                    train_dataset,
                    val_dataset,
                    exp_dir,
                    experiment_config.training,
                    experiment_config.loss,
                    device=device)

    print("Evaluating model...")
    log.info('Evaluating Model')
    evaluation_metrics = evaluate(model, val_dataset, exp_dir)
    log.info(f'Evaluation complete: \n{evaluation_metrics}')

    return {
        'metrics': evaluation_metrics,
        'history': history,
        'config': experiment_config
    }, exp_dir


def run_experiment_suite(
    suite_name,
    output_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    repeats: int = 1,
    shap_interpret: bool = False,
) -> Path:
    """
    Run a predefined experiment suite.

    Args:
        suite_name: name of experiment suite from EXPERIMENT_SUITES
        output_dir: path to save results
        device: torch device
        repeats: number of times to repeat each experiment

    Returns:
        results: list of experiment results
    """
    if suite_name not in EXPERIMENT_SUITES:
        raise ValueError(f"Unknown experiment suite: {suite_name}")

    experiments = EXPERIMENT_SUITES[suite_name] * repeats

    print(highlight_str(f"Running experiment suite: {suite_name}"))
    print(f"Total experiments: {len(experiments)}")

    log.info(
        f'Running Expeirment Suite {suite_name}, {len(experiments)} experiments'
    )

    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    output_dir = Path(output_dir) / f"{suite_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, exp_config in enumerate(experiments):
        print(highlight_str(f"Experiment {i+1}/{len(experiments)}"))
        result, exp_dir = run_single_experiment(exp_config, output_dir, device)
        results.append(result)
        if shap_interpret:
            interpret(exp_dir, device=device, num_segments=8)

    plot_experiment_results(output_dir, ignore_params=['feature_combo'])
    # plot_confusion_matrices(output_dir)
    # plot_experiment_losses(output_dir)
    return output_dir


def evaluate_suite(experiment_dir: Path) -> None:
    plot_experiment_results(experiment_dir, ignore_params=['feature_combo'])
    # plot_confusion_matrices(experiment_dir)
    # plot_experiment_losses(experiment_dir)
