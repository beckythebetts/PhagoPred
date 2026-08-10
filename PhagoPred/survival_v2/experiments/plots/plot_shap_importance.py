"""Cross-model SHAP importance figures: one row per experiment.

Each row is one model: its SHAP heatmap, then the temporal and feature profiles
with the ground truth drawn alongside for reference. Two flavours:

* ``plot_shap_samples_across_models`` — one figure per sample, signed values
* ``plot_shap_average_across_models`` — mean |SHAP| over the dataset

``normalise`` (default) rescales each estimator's temporal/feature values to sum
to 1 so models and ground truth are compared on *relative* importance — a
confident, high-variance model (or a mask-baseline vs. distributional baseline
mismatch) otherwise dominates by scale. Pass ``normalise=False`` for raw magnitudes.

A suite generally mixes scenarios as well as models, and the ground truth is a
property of the *scenario*. Rows are therefore only ever stacked within one
scenario, and each function returns one figure per scenario (a tuple, which
``_plot_and_save`` writes out as ``<name>_1.png``, ``<name>_2.png``, ...).
Sample indices likewise only line up within a scenario, since each scenario
draws its own landmark frames.
"""
from __future__ import annotations
from dataclasses import asdict, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.importance_data import (
    dataset_average,
    horizon_outputs,
    load_sample_importances,
    read_root_attrs,
    sample_indices,
)
from PhagoPred.survival_v2.interpret.importance_plots import (
    feature_panel,
    heatmap_panel,
    outputs_panel,
    series_rows,
    temporal_panel,
)
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()

_SHAP_FILE = 'shap_samples.h5'


def _norm(values: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Rescale so |values| sum to 1, for cross-estimator comparison.

    Each estimator (model, interventional GT, observational GT) is normalised
    independently, so bars/curves show *relative* importance and a scale
    mismatch between them (e.g. mask-baseline vs. distributional-baseline
    KernelSHAP) no longer swamps the shared axis. All-zero / all-NaN inputs are
    returned unchanged rather than divided by zero.
    """
    values = np.asarray(values, dtype=float)
    if not normalise:
        return values
    total = np.nansum(np.abs(values))
    return values / total if total > 0 else values


def _model_label(record: ExperimentRecord, varying_params: dict) -> str:
    """Name a model by whatever config fields differ across the suite."""
    cfg = record.experiemnt_cfg
    if not varying_params:
        return Path(record.experiment_dir).name
    parts = []
    for name in varying_params:
        value = getattr(cfg, name, None)
        parts.append(f'{name}={getattr(value, "name", value)}')
    return ', '.join(parts)


def _shap_path(record: ExperimentRecord) -> Path | None:
    if record.experiment_dir is None:
        return None
    path = Path(record.experiment_dir) / _SHAP_FILE
    return path if path.is_file() else None


def _by_scenario(
        experiments: list[ExperimentRecord]
) -> dict[str, list[ExperimentRecord]]:
    """Group experiments by scenario, dropping those with no usable samples.

    ``compare_importance`` creates shap_samples.h5 before it has any samples to
    put in it, so a suite mid-run contains empty files; those are skipped rather
    than raising out of ``dataset_average``.
    """
    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in experiments:
        path = _shap_path(record)
        if path is None:
            continue
        if not sample_indices(path):
            log.warning(f'{path} has no samples yet; skipping')
            continue
        scenario = read_root_attrs(path)['scenario']
        grouped.setdefault(scenario, []).append(record)
    return grouped


def _row(
    axes,
    label: str,
    heat_matrix: np.ndarray,
    heat_rows: list[str],
    n_frames: int,
    heat_title: str,
    temporal_series: dict,
    feature_series: dict,
    diverging: bool,
    support: np.ndarray | None = None,
) -> None:
    heatmap_panel(axes[0], heat_matrix, heat_rows, n_frames, heat_title,
                  diverging=diverging)
    temporal_panel(axes[1], temporal_series, 'Temporal importance',
                   n_frames=n_frames, support=support)
    feature_panel(axes[2], feature_series, 'Feature importance')
    axes[0].set_ylabel(label, fontsize=9, fontweight='bold')


def _plot_sample_rows(records: list[ExperimentRecord], varying_params: dict,
                      scenario: str, sample_idx: int,
                      normalise: bool = True) -> plt.Figure:
    """One sample of one scenario, one row per model. Signed SHAP.

    The trailing outputs column compares each model's prediction for this sample
    against the ground truth implied by the graph's realised hazard, so a row's
    attributions can be read knowing whether that model got the sample right.
    """
    fig, axes = plt.subplots(len(records), 4,
                             figsize=(25, 3.6 * len(records)), squeeze=False)
    for row, record in enumerate(records):
        h5_path = Path(record.experiment_dir) / _SHAP_FILE
        sample = load_sample_importances(h5_path, sample_idx)
        root = read_root_attrs(h5_path)

        # Drawn before the model-SHAP guard: a model with no SHAP run yet still
        # has a prediction worth showing.
        outputs_panel(
            axes[row, 3],
            root['output_type'],
            ground_truth=horizon_outputs(sample.horizon_hazard,
                                         root['output_type'],
                                         root['hazard_bins']),
            model_prediction=sample.model_prediction,
            hazard_bins=root['hazard_bins'],
            horizon=root['horizon'],
            death_offset=(None if sample.death_frame is None else
                          sample.death_frame - sample.landmark_frame),
        )

        if sample.model_map is None:
            axes[row, 0].text(0.5, 0.5, 'no model SHAP', ha='center',
                              va='center', transform=axes[row, 0].transAxes)
            continue

        temporal_series = {
            key: _norm(sample.ground_truth_temporal(key), normalise)
            for key in sample.ground_truth
        }
        temporal_series['Model'] = _norm(sample.model_temporal_from_map(),
                                         normalise)

        feature_series = {
            key: (sample.feature_names,
                  _norm(sample.ground_truth_feature(key), normalise))
            for key in sample.ground_truth
        }
        feature_series['Model'] = (
            sample.model_feature_names,
            _norm(sample.model_feature_from_map(), normalise))

        _row(axes[row], _model_label(record, varying_params), sample.model_map,
             sample.model_feature_names, sample.landmark_frame,
             'Model KernelSHAP (per-frame)', temporal_series, feature_series,
             diverging=True)

    fig.suptitle(
        f'Model SHAP vs ground truth — {scenario}, sample {sample_idx} '
        f'(lf={sample.landmark_frame})',
        fontweight='bold', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_shap_samples_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        num_plot_samples: int = 5,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per (scenario, sample), rows = models sharing that scenario."""
    grouped = _by_scenario(experiments)
    if not grouped:
        log.warning('No experiments with usable shap_samples.h5; '
                    'skipping SHAP sample plots')
        return None

    figs = []
    for scenario, records in grouped.items():
        # only samples present in every model of this scenario can be compared
        common = set(sample_indices(_shap_path(records[0])))
        for record in records[1:]:
            common &= set(sample_indices(_shap_path(record)))
        indices = sorted(common)[:num_plot_samples]
        if not indices:
            log.warning(f'{scenario}: no sample indices common to all models')
            continue
        for sample_idx in indices:
            figs.append(
                _plot_sample_rows(records, varying_params, scenario,
                                  sample_idx, normalise=normalise))
    return tuple(figs) if figs else None


def plot_shap_average_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per scenario: dataset-average mean |SHAP|, one row per model."""
    grouped = _by_scenario(experiments)
    if not grouped:
        log.warning('No experiments with usable shap_samples.h5; '
                    'skipping SHAP average plots')
        return None

    figs = []
    for scenario, records in grouped.items():
        fig, axes = plt.subplots(len(records), 3,
                                 figsize=(19, 3.6 * len(records)),
                                 squeeze=False)
        for row, record in enumerate(records):
            average = dataset_average(_shap_path(record))
            if 'Model' not in average.maps:
                axes[row, 0].text(0.5, 0.5, 'no model SHAP', ha='center',
                                  va='center',
                                  transform=axes[row, 0].transAxes)
                continue

            feature_series = {
                key: (series_rows(key, average.feature_names,
                                  average.model_feature_names),
                      _norm(values, normalise))
                for key, values in average.feature.items()
            }
            temporal_series = {
                key: _norm(values, normalise)
                for key, values in average.temporal.items()
            }
            _row(axes[row], _model_label(record, varying_params),
                 average.maps['Model'], average.model_feature_names,
                 average.max_landmark_frame,
                 f'Model mean |SHAP| ({average.num_samples} samples)',
                 temporal_series, feature_series, diverging=False,
                 support=average.support)

        fig.suptitle(f'Dataset-average |SHAP| vs ground truth — {scenario}',
                     fontweight='bold', fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figs.append(fig)
    return tuple(figs)
