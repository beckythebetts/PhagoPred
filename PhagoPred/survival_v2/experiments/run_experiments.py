from __future__ import annotations
from pathlib import Path
import json
from datetime import datetime
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

# from PhagoPred.survival_v2.configs.experiment_config import (MODELS, ATTENTION,
#                                                              LOSSES, DATASETS,
#                                                              TRAINING,
#                                                              EXPERIMENT_SUITES,
#                                                              FEATURE_COMBOS)
from PhagoPred.survival_v2.configs.experiments import EXPERIMENT_SUITES, ExperimentCfg, ModelCfg, AttentionCfg, DatasetCfg
from PhagoPred.survival_v2.configs.io import load_experiment_cfg, save_experiment_cfg
from PhagoPred.survival_v2.configs.models import LSTMCfg, CNNCfg, RSFCfg
from PhagoPred.survival_v2.configs.datasets import SurvivalDatasetCfg, BinaryDatasetCfg, BinaryClassDatasetCfg
from PhagoPred.survival_v2.configs.calibration import (
    CalibrationCfg,
    TemperatureScalingCfg,
    VectorScalingCfg,
    PlattScalingCfg,
    IsotonicScalingCfg,
)
from PhagoPred.survival_v2.probability_calib import (
    fit_temperature_scaling,
    fit_vector_scaling,
    fit_platt_scaling,
    fit_isotonic_scaling,
)
from PhagoPred.survival_v2.models.build import build_model
from PhagoPred.survival_v2.models.classical_base import ClassicalSurvivalModel
from PhagoPred.survival_v2.train.train import train
from PhagoPred.survival_v2.evaluate import evaluate
from PhagoPred.survival_v2.data import (
    CellDataset,
    SurvivalCellDataset,
    BinaryCellDataset,
    BinaryClassDataset,
    binary_collate_fn,
    survival_collate_fn,
    binary_class_collate_fn,
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


def _bins_cache_key(dataset_config: SurvivalDatasetCfg) -> str:
    """Cache key shared across learning-curve experiments.

    train_paths varies with training-set size, so it is deliberately excluded;
    the key pins the binning-determining params plus the (fixed) val source so
    genuinely different binnings stay separate while same-suite experiments match.
    """
    return (f"val={dataset_config.val_paths}|"
            f"num_bins={dataset_config.num_bins}|"
            f"max_ttd={dataset_config.max_time_to_death}")


def _load_cached_bins(dataset_config: SurvivalDatasetCfg,
                      cache_dir: Path | None) -> list | None:
    """Return cached event-time bin edges for this config, or None."""
    if cache_dir is None:
        return None
    path = Path(cache_dir) / 'bins.json'
    if not path.exists():
        return None
    with path.open('r') as f:
        return json.load(f).get(_bins_cache_key(dataset_config))


def _store_cached_bins(dataset_config: SurvivalDatasetCfg,
                       cache_dir: Path | None, bins: list) -> None:
    """Persist bin edges, preserving any existing entries in the cache file."""
    if cache_dir is None:
        return
    path = Path(cache_dir) / 'bins.json'
    store = {}
    if path.exists():
        with path.open('r') as f:
            store = json.load(f)
    store[_bins_cache_key(dataset_config)] = list(bins)
    with path.open('w') as f:
        json.dump(store, f, indent=2)


def build_datasets(
        dataset_config: DatasetCfg,
        feature_combo: list,
        is_classical: bool = False,
        cache_dir: Path | None = None) -> tuple[CellDataset, CellDataset]:
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
    calibrate_dataset = None
    if isinstance(dataset_config, BinaryDatasetCfg):

        dataset_args = {
            'feature_names': feature_combo,
            'min_length': dataset_config.min_length,
            'model_type': model_type,
            'prediction_horizon': dataset_config.prediction_horizon,
            'num_cells': dataset_config.num_cells,
        }
        train_dataset = BinaryCellDataset(
            hdf5_paths=dataset_config.train_paths, **dataset_args)

        means, stds = train_dataset.get_normalisation_stats()

        val_dataset = BinaryCellDataset(hdf5_paths=dataset_config.val_paths,
                                        means=means,
                                        stds=stds,
                                        **dataset_args)

        if dataset_config.calibrate_paths is not None:
            calibrate_dataset = BinaryCellDataset(
                hdf5_paths=dataset_config.calibrate_paths,
                means=means,
                stds=stds,
                **dataset_args,
            )

    elif isinstance(dataset_config, SurvivalDatasetCfg):
        dataset_args = {
            'feature_names': feature_combo,
            'min_length': dataset_config.min_length,
            'model_type': model_type,
            'num_bins': dataset_config.num_bins,
            'max_time_to_death': dataset_config.max_time_to_death,
        }

        event_time_bins = _load_cached_bins(dataset_config, cache_dir)
        if event_time_bins is not None:
            train_dataset = SurvivalCellDataset(
                hdf5_paths=dataset_config.train_paths,
                event_time_bins=event_time_bins,
                **dataset_args)
        else:
            train_dataset = SurvivalCellDataset(
                hdf5_paths=dataset_config.train_paths, **dataset_args)
            event_time_bins = train_dataset.get_bins().tolist()
            _store_cached_bins(dataset_config, cache_dir, event_time_bins)

        means, stds = train_dataset.get_normalisation_stats()

        val_dataset = SurvivalCellDataset(
            hdf5_paths=dataset_config.val_paths,
            means=means,
            stds=stds,
            event_time_bins=event_time_bins,
            **dataset_args,
        )

        if dataset_config.calibrate_paths is not None:
            calibrate_dataset = SurvivalCellDataset(
                hdf5_paths=dataset_config.calibrate_paths,
                means=means,
                stds=stds,
                event_time_bins=event_time_bins,
                **dataset_args,
            )

        dataset_config.bins = list(event_time_bins)

    elif isinstance(dataset_config, BinaryClassDatasetCfg):
        dataset_args = {
            'feature_names': feature_combo,
            'min_length': dataset_config.min_length,
            'model_type': model_type,
            'class_dict': dataset_config.class_dict,
        }

        train_dataset = BinaryClassDataset(
            hdf5_paths=dataset_config.train_paths,
            **dataset_args,
        )
        means, stds = train_dataset.get_normalisation_stats()
        val_dataset = BinaryClassDataset(
            hdf5_paths=dataset_config.val_paths,
            means=means,
            stds=stds,
            **dataset_args,
        )
        if dataset_config.calibrate_paths is not None:
            calibrate_dataset = BinaryClassDataset(
                hdf5_paths=dataset_config.calibrate_paths,
                means=means,
                stds=stds,
                **dataset_args,
            )
    if not is_classical:
        train_dataset.apply_normalisation()
        val_dataset.apply_normalisation()
        if calibrate_dataset is not None:
            calibrate_dataset.apply_normalisation()

    dataset_config.normalisation_means = means.tolist()
    dataset_config.normalisation_stds = stds.tolist()

    return train_dataset, val_dataset, calibrate_dataset


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
    train_dataset, val_dataset, calibrate_dataset = build_datasets(
        experiment_config.dataset,
        experiment_config.feature_combo,
        is_classical=experiment_config.model.is_classical,
        cache_dir=output_dir)

    if isinstance(train_dataset, BinaryCellDataset):

        pos_weight = train_dataset.get_pos_weight()
        setattr(experiment_config.loss, 'pos_weight', pos_weight)
        log.info(f"Computed pos_weight: {pos_weight:.2f}")

    elif isinstance(train_dataset, SurvivalCellDataset):
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

    save_experiment_cfg(experiment_config, exp_dir / 'config.json')

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

    if experiment_config.calibration is not None and experiment_config.dataset.calibrate_paths:
        print('Fitting calibration...')
        cal_dataset = _build_cal_dataset(experiment_config)
        _fit_and_store_calibration(experiment_config, model, cal_dataset,
                                   device)
        save_experiment_cfg(experiment_config, exp_dir / 'config.json')
        log.info(f'Calibration fitted: {experiment_config.calibration}')

    print("Evaluating model...")
    log.info('Evaluating Model')
    evaluation_metrics = evaluate(model, val_dataset, exp_dir,
                                  experiment_config.dataset, device)
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
    # """
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
            interpret(exp_dir, device=device, num_smaples=1000)

    plot_experiment_results(output_dir, ignore_params=['feature_combo'])
    # plot_confusion_matrices(output_dir)
    # plot_experiment_losses(output_dir)
    return output_dir


def _build_cal_dataset(cfg: ExperimentCfg) -> CellDataset:
    """Build the calibration dataset using normalisation stats from the config."""
    is_classical = cfg.model.is_classical
    model_type = 'classical' if is_classical else 'deep'
    means = cfg.dataset.normalisation_means
    stds = cfg.dataset.normalisation_stds

    if isinstance(cfg.dataset, BinaryDatasetCfg):
        ds = BinaryCellDataset(
            hdf5_paths=cfg.dataset.calibrate_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            prediction_horizon=cfg.dataset.prediction_horizon,
            means=means,
            stds=stds,
        )
    elif isinstance(cfg.dataset, SurvivalDatasetCfg):
        ds = SurvivalCellDataset(
            hdf5_paths=cfg.dataset.calibrate_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            num_bins=cfg.dataset.num_bins,
            max_time_to_death=cfg.dataset.max_time_to_death,
            event_time_bins=cfg.dataset.bins,
            means=means,
            stds=stds,
        )
    elif isinstance(cfg.dataset, BinaryClassDatasetCfg):
        ds = BinaryClassDataset(
            hdf5_paths=cfg.dataset.calibrate_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            class_dict=cfg.dataset.class_dict,
            means=means,
            stds=stds,
        )
    else:
        raise TypeError(f'Unknown dataset config type: {type(cfg.dataset)}')

    if not is_classical:
        ds.apply_normalisation()
    return ds


def _get_calibration_inputs(model,
                            dataset: CellDataset,
                            device: str,
                            n_passes: int = 10):
    """Extract logits/predictions and labels from a dataset for calibration fitting."""
    is_classical = isinstance(model, ClassicalSurvivalModel)

    if is_classical:
        tabular = model.get_temporal_summary(dataset)
        predictions = model._predict(tabular.temporal_summary_features)
        if isinstance(dataset, BinaryCellDataset):
            logits = predictions[:, 0]
            labels = tabular.event.astype(int)
            return logits, labels, None, None
        else:
            pmf = predictions
            bins = tabular.time_to_event_bin.astype(int)
            events = tabular.event_indicator.astype(int)
            return pmf, None, bins, events
    else:
        import numpy as np
        model.eval()
        model = model.to(device)
        all_outputs, all_bins, all_events, all_labels = [], [], [], []

        if isinstance(dataset, BinaryCellDataset):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=lambda b: binary_collate_fn(b, device=device))
            with torch.no_grad():
                for _ in range(n_passes):
                    for batch in loader:
                        out = model(batch.features.to(device),
                                    batch.length.to(device),
                                    mask=batch.mask.to(device)
                                    if batch.mask is not None else None)
                        all_outputs.append(out.cpu().numpy())
                        all_labels.append(batch.event.cpu().numpy())
            return (np.concatenate(all_outputs).squeeze(),
                    np.concatenate(all_labels), None, None)
        elif isinstance(dataset, BinaryClassDataset):
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=lambda b: binary_class_collate_fn(b, device=device))
            with torch.no_grad():
                for _ in range(n_passes):
                    for batch in loader:
                        out = model(batch.features.to(device),
                                    batch.length.to(device),
                                    mask=batch.mask.to(device)
                                    if batch.mask is not None else None)
                        all_outputs.append(out.cpu().numpy())
                        all_labels.append(batch.target_class.cpu().numpy())
            return (np.concatenate(all_outputs).squeeze(),
                    np.concatenate(all_labels), None, None)
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=lambda b: survival_collate_fn(b, device=device))
            with torch.no_grad():
                for _ in range(n_passes):
                    for batch in loader:
                        out = model(batch.features.to(device),
                                    batch.length.to(device),
                                    mask=batch.mask.to(device))
                        all_outputs.append(out.cpu().numpy())
                        all_bins.append(batch.time_to_event_bin.cpu().numpy())
                        all_events.append(batch.event_indicator.cpu().numpy())
            return (np.concatenate(all_outputs), None,
                    np.concatenate(all_bins), np.concatenate(all_events))


def _fit_and_store_calibration(cfg: ExperimentCfg, model, dataset: CellDataset,
                               device: str) -> None:
    """Fit calibration, store fitted params on cfg, apply to model."""
    is_classical = isinstance(model, ClassicalSurvivalModel)
    cal_cfg = cfg.calibration
    logits_or_pmf, labels, bins, events = _get_calibration_inputs(
        model, dataset, device)

    if isinstance(cal_cfg, PlattScalingCfg):
        result = fit_platt_scaling(logits_or_pmf,
                                   labels,
                                   from_logits=not is_classical)
        cal_cfg.a = result.a
        cal_cfg.b = result.b
    elif isinstance(cal_cfg, IsotonicScalingCfg):
        if labels is None:
            raise ValueError(
                "Isotonic calibration is binary-only; got a survival dataset.")
        result = fit_isotonic_scaling(logits_or_pmf,
                                      labels,
                                      from_logits=not is_classical)
        cal_cfg.x_thresholds = result.x_thresholds.tolist()
        cal_cfg.y_thresholds = result.y_thresholds.tolist()
    elif isinstance(cal_cfg, TemperatureScalingCfg):
        result = fit_temperature_scaling(logits_or_pmf,
                                         bins,
                                         events,
                                         from_logits=not is_classical)
        cal_cfg.T = result.T
    elif isinstance(cal_cfg, VectorScalingCfg):
        result = fit_vector_scaling(logits_or_pmf,
                                    bins,
                                    events,
                                    from_logits=not is_classical)
        cal_cfg.w = result.w.tolist()
        cal_cfg.b = result.b.tolist()
    else:
        return

    model.set_calibration(result)


def _build_val_dataset(cfg: ExperimentCfg) -> CellDataset:
    """Rebuild the validation dataset from a saved experiment config.

    Uses the normalisation stats and bins already stored in the config,
    so no training data is needed.
    """
    is_classical = cfg.model.is_classical
    model_type = 'classical' if is_classical else 'deep'
    means = cfg.dataset.normalisation_means
    stds = cfg.dataset.normalisation_stds

    if isinstance(cfg.dataset, BinaryDatasetCfg):
        val_dataset = BinaryCellDataset(
            hdf5_paths=cfg.dataset.val_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            prediction_horizon=cfg.dataset.prediction_horizon,
            means=means,
            stds=stds,
        )
    elif isinstance(cfg.dataset, SurvivalDatasetCfg):
        val_dataset = SurvivalCellDataset(
            hdf5_paths=cfg.dataset.val_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            num_bins=cfg.dataset.num_bins,
            max_time_to_death=cfg.dataset.max_time_to_death,
            event_time_bins=cfg.dataset.bins,
            means=means,
            stds=stds,
        )
    elif isinstance(cfg.dataset, BinaryClassDatasetCfg):
        val_dataset = BinaryClassDataset(
            hdf5_paths=cfg.dataset.val_paths,
            feature_names=cfg.feature_combo,
            min_length=cfg.dataset.min_length,
            model_type=model_type,
            class_dict=cfg.dataset.class_dict,
            means=means,
            stds=stds,
        )
    else:
        raise TypeError(f'Unknown dataset config type: {type(cfg.dataset)}')

    if not is_classical:
        val_dataset.apply_normalisation()
    return val_dataset


def _load_model(cfg: ExperimentCfg, model_path: Path,
                device: str) -> 'SurvivalModel | ClassicalSurvivalModel':
    """Load a saved model from disk using its experiment config."""
    from PhagoPred.survival_v2.models.classical_base import ClassicalSurvivalModel
    num_features = len(cfg.feature_combo)
    num_bins = cfg.dataset.num_bins
    model = build_model(cfg.model,
                        cfg.attention,
                        num_features,
                        num_bins,
                        device,
                        feature_names=cfg.feature_combo)
    if isinstance(model, ClassicalSurvivalModel):
        model = model.__class__.load(str(model_path))
    else:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
    return model


def evaluate_suite(experiment_dir: Path,
                   device: str = 'cpu',
                   replot_only: bool = False) -> None:
    """Re-evaluate all experiments in a suite directory without retraining.

    For each experiment sub-directory:
      - loads config.json
      - rebuilds the validation dataset using saved normalisation stats
      - loads the saved model weights
      - runs evaluation and overwrites evaluation_results.json
    Then replots the suite results.

    Args:
        experiment_dir: path to the suite directory (contains experiment_XX/ subdirs)
        device: torch device to use for deep models
        replot_only: if True, skip re-evaluation and just replot existing results
    """
    experiment_dir = Path(experiment_dir)
    exp_dirs = list(sorted(p for p in experiment_dir.iterdir() if p.is_dir()))

    for i, exp_dir in enumerate(exp_dirs):
        print(f'=== EVALUATING EXPERIRIMENT {i+1} / {len(exp_dirs)} ===')
        config_path = exp_dir / 'config.json'
        model_path = exp_dir / 'model.pkl'

        if not config_path.exists():
            log.warning(f'No config.json in {exp_dir}, skipping')
            continue
        if not model_path.exists():
            log.warning(f'No model.pkl in {exp_dir}, skipping')
            continue

        if replot_only:
            continue

        log.info(f'Re-evaluating {exp_dir.name}')
        cfg = load_experiment_cfg(config_path)

        val_dataset = _build_val_dataset(cfg)
        model = _load_model(cfg, model_path, device)

        if cfg.calibration is not None:
            result = cfg.calibration.to_result()
            if result is not None:
                model.set_calibration(result)
                log.info(f'Restored calibration: {cfg.calibration.name}')

        evaluate(model, val_dataset, exp_dir, cfg.dataset, device)
        log.info(f'Evaluation complete for {exp_dir.name}')

    plot_experiment_results(experiment_dir, ignore_params=['feature_combo'])


def test_variances(exp_dir: Path):
    config_path = exp_dir / 'config.json'
    cfg = load_experiment_cfg(config_path)
    val_dataset = _build_val_dataset(cfg)
    print(val_dataset.get_total_variances(), val_dataset._compute_variance())
