from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import json
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py
from tqdm import tqdm

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.utils.io import load_model, load_dataset
from PhagoPred.survival_v2.configs import ExperimentCfg
from PhagoPred.survival_v2.configs.models import RSFCfg
from PhagoPred.survival_v2.configs.losses import BinaryLossCfg
from PhagoPred.survival_v2.models import SurvivalModel
from PhagoPred.survival_v2.data import (
    CellDataset,
    CellSample,
    BinaryCellDataset,
    BinaryClassDataset,
    SurvivalCellDataset,
)
from .SHAP_kernel import KernelSHAP, KernelSHAPResults
from .importance_plots import (
    MODEL_ONLY_PANELS,
    plot_dataset_average,
    plot_samples,
)

log = get_logger()


@dataclass
class RSFImportanceResults:
    """Feature and temporal importance results for RSF models."""
    feature_importance: np.ndarray  # (n_groups,) summed over all windows
    feature_names: list[
        str]  # e.g. ['mean_frame_count', 'std_frame_count', ...]
    temporal_importance_by_scale: dict  # {ws: np.ndarray (num_windows,)}
    window_starts_by_scale: dict  # {ws: np.ndarray (num_windows,)}


OUTPUT_TYPE = {
    BinaryClassDataset: 'binary',
    BinaryCellDataset: 'binary',
    SurvivalCellDataset: 'expected_time'
}


def interpret(
    experiment_dir: Path | str,
    num_samples: int = 10,
    num_shap_samples: int | None = None,
    num_plt_samples: int = 10,
    num_background_samples: int = 8,
    num_segments: int = 50,
) -> None:
    log.info(f'Starting interpret on {experiment_dir}')
    experiment_dir = Path(experiment_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Using device {device}')

    # === GET MODEL / CONFIG ===
    model, cfg, checkpoint = load_model(experiment_dir, device)

    dataset = load_dataset(experiment_dir, 'val')
    hazard_bins = None
    if hasattr(dataset, 'event_time_bins'):
        hazard_bins = dataset.event_time_bins

    model_feat_names: list[str] = cfg.feature_combo
    means, stds = np.array(checkpoint.get('normalalisation_means'),
                           dtype=np.float32), np.array(
                               checkpoint.get('normalalisation_stds'),
                               dtype=np.float32)

    kernel_shap = KernelSHAP(model, model_feat_names, device)

    results, batch = kernel_shap.analyse_batch(
        dataset,
        device=device,
        num_samples=num_samples,
        num_segments=num_segments,
        nsamples_temporal_feature=num_shap_samples,
        output_type=OUTPUT_TYPE[type(dataset)],
        time_bins=hazard_bins,
        return_batch=True)

    _create_shap_file(experiment_dir / 'SHAP.h5', results, batch,
                      dataset.feature_names)

    plot_interpret(experiment_dir, num_plt_samples)


def _create_shap_file(path: Path, results: KernelSHAPResults, batch: dict,
                      feature_names: list[str]) -> None:
    log.info(f'Writing SHAP reuslts to {path}')
    with h5py.File(path, 'w') as f:
        for i in tqdm(range(len(batch['landmark_frame'])),
                      desc='Writing SHAP results'):

            group = f.create_group(str(i))
            group.attrs['Features'] = feature_names
            group.attrs['Landmark Frame'] = batch['landmark_frame'][i]

            if 'death_frame' in batch.keys():
                group.attrs['Death Frame'] = batch['death_frame'][i]

            if 'target_class' in batch.keys():
                group.attrs['Target Class'] = batch['target_class'].cpu()[i]
            # (feature, frame) to match importance_data.load_sample_importances,
            # which slices Signals as ``[:, :lf]``. The collated tensor is
            # (sample, frame, feature).
            group.create_dataset('Signals',
                                 data=np.asarray(batch['features'].cpu()[i]).T,
                                 dtype=float)
            _write_model_shap_results(f, i, results, feature_names)


def _write_model_shap_results(h5_file: h5py.File, sample_idx: int,
                              shap_results: KernelSHAPResults,
                              model_featurre_names: list[str]) -> None:
    # analyse_batch aggregates every sample into these arrays (sample-major), so
    # each group is indexed by sample_idx — not 0, which would store sample 0's
    # map under every group.
    f = h5_file
    sample = f[str(sample_idx)]
    model = sample.create_dataset(
        'Model',
        data=shap_results.temporal_feature_shap_values[sample_idx],
        dtype=float)
    model.attrs['Temporal'] = shap_results.temporal_shap_values[sample_idx]
    model.attrs['Feature'] = shap_results.feature_shap_values[sample_idx]
    model.attrs['Segment Boundaries'] = shap_results.segment_boundaries
    model.attrs['Feature Names'] = model_featurre_names


# ─────────────────────────── plotting ───────────────────────────
# Same figures as ground_truth_importance, minus the ground-truth panels: there
# is no causal graph behind a real dataset, so SHAP.h5 stores model SHAP only.
# The assembly is shared from importance_plots; MODEL_ONLY_PANELS drops the
# interventional / observational / outputs axes.


def plot_interpret(experiment_dir: Path | str,
                   num_plot_samples: int = 10) -> None:
    """Write per-sample and dataset-average model-SHAP figures for SHAP.h5."""
    experiment_dir = Path(experiment_dir)
    h5_path = experiment_dir / 'SHAP.h5'
    if not h5_path.is_file():
        log.warning(f'{h5_path} not found; skipping interpret plots')
        return

    save_dir = experiment_dir / 'shap'
    plot_samples(h5_path,
                 save_dir,
                 num_plot_samples,
                 title_prefix=experiment_dir.name,
                 panels=MODEL_ONLY_PANELS)

    average_fig = plot_dataset_average(h5_path, title=experiment_dir.name)
    average_path = save_dir / 'dataset_average.png'
    average_fig.savefig(average_path, bbox_inches='tight', dpi=120)
    plt.close(average_fig)
    log.info(f'Saved interpret figures to {save_dir}')
