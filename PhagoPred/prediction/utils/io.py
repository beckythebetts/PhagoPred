from pathlib import Path
from typing import Literal, Optional, Union
import logging

import torch
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.configs.io import load_experiment_cfg
from PhagoPred.survival_v2.configs.models import RSFCfg
from PhagoPred.survival_v2.configs.datasets import BinaryDatasetCfg, BinaryClassDatasetCfg
from PhagoPred.survival_v2.configs.experiments import ExperimentCfg
from PhagoPred.survival_v2.models.base import SurvivalModel
from PhagoPred.survival_v2.models.classical_base import ClassicalSurvivalModel
from PhagoPred.survival_v2.data import CellDataset, SurvivalCellDataset, BinaryCellDataset, BinaryClassDataset
from PhagoPred.survival_v2.models.build import build_model

log = get_logger()


def _restore_calibration(model, cfg: ExperimentCfg) -> None:
    """Restore post-hoc calibration saved in the experiment config onto the model.

    Mirrors the re-evaluation path in run_experiments so that models loaded for
    interpretation/analysis produce the same calibrated probabilities as the
    evaluation pipeline (otherwise predictions are the raw, uncalibrated logits).
    """
    cal_cfg = getattr(cfg, "calibration", None)
    if cal_cfg is None:
        return
    result = cal_cfg.to_result()
    if result is not None:
        model.set_calibration(result)
        log.info(f"Restored calibration: {cal_cfg.name}")


def load_model(
    experiment_dir: Union[str, Path],
    device: str = "cpu",
) -> tuple[Union[SurvivalModel, ClassicalSurvivalModel], ExperimentCfg, dict]:
    """
    Load a trained model from an experiment directory.

    Expects the directory to contain:
        config.json  — experiment configuration (always present)
        model.pkl    — saved model weights or pickled classical model

    Args:
        experiment_dir: Path to the experiment directory.
        device: Device to load deep models onto ('cpu', 'cuda', etc.).

    Returns:
        model: Loaded SurvivalModel or ClassicalSurvivalModel.
        cfg: The ExperimentCfg used to train the model.
        checkpoint: For deep models, the raw checkpoint dict (contains
                    'normalisation_means' and 'normalisation_stds').
                    For classical models, an empty dict.
    """
    log.info(f'Loading model from {experiment_dir} to {device}')
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "config.json"
    model_path = experiment_dir / "model.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {experiment_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found in {experiment_dir}")

    cfg = load_experiment_cfg(config_path)

    num_features = len(cfg.feature_combo)
    num_bins = cfg.dataset.num_bins

    if isinstance(cfg.model, RSFCfg):
        log.info('Building Random Forest Model')
        model = ClassicalSurvivalModel.load(str(model_path))
        _restore_calibration(model, cfg)
        return model, cfg, {}

    # Deep model: reconstruct architecture then load weights
    log.info(f'Building model {cfg.model.model_type}')
    model = build_model(
        cfg.model,
        cfg.attention,
        num_features,
        num_bins,
        device,
        feature_names=cfg.feature_combo,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _restore_calibration(model, cfg)

    return model, cfg, checkpoint


def load_dataset(
    experiment_dir: Union[str, Path],
    split: Literal["train", "val"] = "val",
    hdf5_paths: Optional[list[Union[str, Path]]] = None,
) -> CellDataset:
    """
    Load a single normalized dataset from an experiment directory.

    Reads structural parameters (num_bins, min_length, etc.) and normalisation
    stats from config.json so the dataset is always normalized consistently
    with training, without recomputing stats from scratch.

    Args:
        experiment_dir: Path to the experiment directory containing config.json.
        split: Which paths from the config to use — 'train' or 'val'.
               Ignored when hdf5_paths is provided.
        hdf5_paths: Override the paths entirely (e.g. a held-out test set).

    Returns:
        A normalized CellDataset ready for SHAP analysis or evaluation.
    """

    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "config.json"
    log.info(f'Loading datset from config: {config_path}')

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {experiment_dir}")

    cfg = load_experiment_cfg(config_path)
    dcfg = cfg.dataset

    log.info(f'Datset config: {dcfg}')

    if dcfg.normalisation_means is None or dcfg.normalisation_stds is None:
        raise ValueError("config.json does not contain normalisation stats. ")

    means = np.array(dcfg.normalisation_means)
    stds = np.array(dcfg.normalisation_stds)

    paths = hdf5_paths if hdf5_paths is not None else (
        dcfg.train_paths if split == "train" else dcfg.val_paths)

    is_classical = isinstance(cfg.model, RSFCfg)
    model_type = "classical" if is_classical else "deep"

    common = dict(
        hdf5_paths=paths,
        feature_names=cfg.feature_combo,
        min_length=dcfg.min_length,
        model_type=model_type,
        means=means,
        stds=stds,
    )

    if isinstance(dcfg, BinaryDatasetCfg):
        dataset: CellDataset = BinaryCellDataset(
            prediction_horizon=dcfg.prediction_horizon,
            **common,
        )
    elif isinstance(dcfg, BinaryClassDatasetCfg):
        dataset = BinaryClassDataset(
            class_dict=dcfg.class_dict,
            **common,
        )
    else:
        dataset = SurvivalCellDataset(
            num_bins=dcfg.num_bins,
            max_time_to_death=dcfg.max_time_to_death,
            event_time_bins=dcfg.bins,
            **common,
        )
        # Bin edges are computed at train time and persisted to config.json.
        # Without them every sample's time_to_event_bin is None, which breaks
        # collation. Fall back to equal-width bins for older configs missing them.
        if dcfg.bins is None:
            log.warning(
                "config.json has no event_time_bins; falling back to "
                "equal-width bins. Predictions may not match training.")
            dataset.get_bins()

    if not is_classical:
        dataset.apply_normalisation()

    return dataset


def get_normalisation_stats(
    checkpoint: dict, ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract normalisation means and stds from a deep model checkpoint.

    Args:
        checkpoint: The dict returned as the third element of load_model.

    Returns:
        (means, stds) as numpy arrays, or (None, None) for classical models.
    """
    means = checkpoint.get("normalisation_means")
    stds = checkpoint.get("normalisation_stds")
    return means, stds
