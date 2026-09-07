from __future__ import annotations
from dataclasses import asdict, fields
from pathlib import Path
from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.importance_data import (
    dataset_average,
    horizon_outputs,
    load_sample_importances,
    pad_frames,
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

_SHAP_FILES = ['shap_samples.h5', 'SHAP.h5']


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


def _hashable(value):
    """A hashable stand-in for a config field value.

    Config fields are often nested dataclasses (e.g. a model config), which
    aren't hashable and can't be dict keys directly. ``.name`` (enums) is the
    common case; anything else falls back to its repr, which is stable and
    identical for equal-valued dataclasses since ``dataclasses`` generates
    ``__repr__`` from the fields.
    """
    name = getattr(value, 'name', None)
    if name is not None:
        return name
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)


def _model_key(record: ExperimentRecord, varying_params: dict) -> tuple:
    """Group key for records that are repeats/kfolds of the same model.

    Unlike ``_model_label`` (which falls back to the experiment directory name,
    unique per run), this is constant across repeats of a config so they can be
    averaged together.
    """
    cfg = record.experiemnt_cfg
    return tuple(
        _hashable(getattr(cfg, name, None)) for name in varying_params)


def _group_label(key: tuple, varying_params: dict) -> str:
    if not varying_params:
        return 'Model'
    parts = [f'{name}={value}' for name, value in zip(varying_params, key)]
    return ', '.join(parts)


def _nanmean(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(stack, axis=0)


def _nanstd(stack: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanstd(stack, axis=0)


def _shap_path(record: ExperimentRecord) -> Path | None:
    if record.experiment_dir is None:
        return None
    for shap_file in _SHAP_FILES:
        path = Path(record.experiment_dir) / shap_file
        if path.is_file():
            return path
    log.warning(f'No SHAP file found')
    return None


# def _by_scenario(
#         experiments: list[ExperimentRecord]
# ) -> dict[str, list[ExperimentRecord]]:
#     """Group experiments by scenario, dropping those with no usable samples.

#     ``compare_importance`` creates shap_samples.h5 before it has any samples to
#     put in it, so a suite mid-run contains empty files; those are skipped rather
#     than raising out of ``dataset_average``.
#     """
#     grouped: dict[str, list[ExperimentRecord]] = {}
#     for record in experiments:
#         path = _shap_path(record)
#         if path is None:
#             continue
#         if not sample_indices(path):
#             log.warning(f'{path} has no samples yet; skipping')
#             continue

#         scenario = read_root_attrs(path)['scenario']
#         grouped.setdefault(scenario, []).append(record)
#     return grouped


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
    temporal_title: str = 'Temporal importance',
    feature_title: str = 'Feature importance',
    temporal_err: dict | None = None,
    feature_err: dict | None = None,
) -> None:
    heatmap_panel(axes[0],
                  heat_matrix,
                  heat_rows,
                  n_frames,
                  heat_title,
                  diverging=diverging)
    temporal_panel(axes[1],
                   temporal_series,
                   temporal_title,
                   n_frames=n_frames,
                   support=support,
                   err=temporal_err)
    feature_panel(axes[2], feature_series, feature_title, err=feature_err)
    axes[0].set_ylabel(label, fontsize=9, fontweight='bold')


def _plot_sample_rows(records: list[ExperimentRecord],
                      varying_params: dict,
                      scenario: str,
                      sample_idx: int,
                      normalise: bool = True) -> plt.Figure:
    """One sample of one scenario, one row per model. Signed SHAP.

    The trailing outputs column compares each model's prediction for this sample
    against the ground truth implied by the graph's realised hazard, so a row's
    attributions can be read knowing whether that model got the sample right.
    """
    fig, axes = plt.subplots(len(records),
                             4,
                             figsize=(25, 3.6 * len(records)),
                             squeeze=False)
    for row, record in enumerate(records):
        # h5_path = Path(record.experiment_dir) / _SHAP_FILE
        h5_path = _shap_path(record)
        sample = load_sample_importances(h5_path, sample_idx)
        # root = read_root_attrs(h5_path)

        # Drawn before the model-SHAP guard: a model with no SHAP run yet still
        # has a prediction worth showing.
        ds_cfg = record.experiemnt_cfg.dataset
        if ds_cfg.num_bins == 1:
            output_type = 'binary'
            hazard_bins = None
            if hasattr(ds_cfg, 'prediction_horizon'):
                horizon = ds_cfg.prediction_horizon
            else:
                horizon = 0
        else:
            output_type = 'survival'
            hazard_bins = record.experiemnt_cfg.dataset.num_bins
            horizon = hazard_bins[-1]

        outputs_panel(
            axes[row, 3],
            output_type,
            ground_truth=horizon_outputs(sample.horizon_hazard, output_type,
                                         hazard_bins),
            # ground_truth=horizon_outputs(sample.horizon_hazard,
            #                              record.experiemnt_cfg),
            model_prediction=sample.model_prediction,
            hazard_bins=hazard_bins,
            horizon=horizon,
            death_offset=(None if sample.death_frame is None else
                          sample.death_frame - sample.landmark_frame),
        )

        if sample.model_map is None:
            axes[row, 0].text(0.5,
                              0.5,
                              'no model SHAP',
                              ha='center',
                              va='center',
                              transform=axes[row, 0].transAxes)
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
        feature_series['Model'] = (sample.model_feature_names,
                                   _norm(sample.model_feature_from_map(),
                                         normalise))

        _row(axes[row],
             _model_label(record, varying_params),
             sample.model_map,
             sample.model_feature_names,
             sample.landmark_frame,
             'Model KernelSHAP (per-frame)',
             temporal_series,
             feature_series,
             diverging=True)

    fig.suptitle(
        f'Model SHAP vs ground truth — {scenario}, sample {sample_idx} '
        f'(lf={sample.landmark_frame})',
        fontweight='bold',
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_shap_samples_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        num_plot_samples: int = 5,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per (scenario, sample), rows = models sharing that scenario."""
    # grouped = _by_scenario(experiments)
    # if not grouped:
    #     log.warning('No experiments with usable shap_samples.h5; '
    #                 'skipping SHAP sample plots')
    #     return None
    grouped_by_dataset: defaultdict[str,
                                    list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        grouped_by_dataset[exp.experiemnt_cfg.dataset.name].append(exp)
    figs = []
    for scenario, records in grouped_by_dataset.items():
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
                _plot_sample_rows(records,
                                  varying_params,
                                  scenario,
                                  sample_idx,
                                  normalise=normalise))
    return tuple(figs) if figs else None


def plot_shap_average_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per scenario: dataset-average mean |SHAP|, one row per model."""
    # grouped = _by_scenario(experiments)
    # if not grouped:
    #     log.warning('No experiments with usable shap_samples.h5; '
    #                 'skipping SHAP average plots')
    #     return None

    grouped_by_dataset: defaultdict[str,
                                    list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        grouped_by_dataset[exp.experiemnt_cfg.dataset.name].append(exp)
    figs = []
    for scenario, records in grouped_by_dataset.items():
        fig, axes = plt.subplots(len(records),
                                 3,
                                 figsize=(19, 3.6 * len(records)),
                                 squeeze=False)
        for row, record in enumerate(records):
            average = dataset_average(_shap_path(record))
            if 'Model' not in average.maps:
                axes[row, 0].text(0.5,
                                  0.5,
                                  'no model SHAP',
                                  ha='center',
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
            _row(axes[row],
                 _model_label(record, varying_params),
                 average.maps['Model'],
                 average.model_feature_names,
                 average.max_landmark_frame,
                 f'Model mean |SHAP| ({average.num_samples} samples)',
                 temporal_series,
                 feature_series,
                 diverging=False,
                 support=average.support)

        fig.suptitle(f'Dataset-average |SHAP| vs ground truth — {scenario}',
                     fontweight='bold',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figs.append(fig)
    return tuple(figs)


def plot_shap_fold_average_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per scenario: mean +/- std |SHAP| across folds/repeats, one row per model.

    Companion to ``plot_shap_average_across_models``, which draws each repeat's
    dataset-average as its own row. Here, records sharing the same varying-param
    combination (i.e. repeats of the same model, or kfold splits of it) are
    collapsed into a single row: temporal and feature curves show mean +/- std
    across folds, and the heatmap shows the fold mean only (a spread doesn't
    render on a heatmap).
    """
    grouped_by_dataset: defaultdict[str,
                                    list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        grouped_by_dataset[exp.experiemnt_cfg.dataset.name].append(exp)

    figs = []
    for scenario, records in grouped_by_dataset.items():
        groups: defaultdict[tuple, list[ExperimentRecord]] = defaultdict(list)
        for record in records:
            groups[_model_key(record, varying_params)].append(record)

        fig, axes = plt.subplots(len(groups),
                                 3,
                                 figsize=(19, 3.6 * len(groups)),
                                 squeeze=False)
        for row, (key, group_records) in enumerate(groups.items()):
            label = _group_label(key, varying_params)
            averages = [
                dataset_average(_shap_path(r)) for r in group_records
                if _shap_path(r) is not None
            ]
            averages = [a for a in averages if 'Model' in a.maps]
            if not averages:
                axes[row, 0].text(0.5,
                                  0.5,
                                  'no model SHAP',
                                  ha='center',
                                  va='center',
                                  transform=axes[row, 0].transAxes)
                axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')
                continue

            max_lf = max(a.max_landmark_frame for a in averages)
            feature_names = averages[0].feature_names
            model_feature_names = averages[0].model_feature_names
            keys = set().union(*(a.temporal.keys() for a in averages))

            temporal_mean, temporal_std = {}, {}
            feature_series, feature_err = {}, {}
            map_mean = {}
            for est_key in keys:
                temporal_stack = np.stack([
                    pad_frames(_norm(a.temporal[est_key], normalise), max_lf)
                    for a in averages if est_key in a.temporal
                ])
                temporal_mean[est_key] = _nanmean(temporal_stack)
                temporal_std[est_key] = _nanstd(temporal_stack)

                feature_stack = np.stack([
                    _norm(a.feature[est_key], normalise) for a in averages
                    if est_key in a.feature
                ])
                rows = series_rows(est_key, feature_names, model_feature_names)
                feature_series[est_key] = (rows, feature_stack.mean(axis=0))
                feature_err[est_key] = (rows, feature_stack.std(axis=0))

                map_stack = np.stack([
                    pad_frames(a.maps[est_key], max_lf) for a in averages
                    if est_key in a.maps
                ])
                map_mean[est_key] = _nanmean(map_stack)

            support = np.nan_to_num(
                np.stack([pad_frames(a.support, max_lf)
                          for a in averages])).sum(axis=0)

            _row(
                axes[row],
                label,
                map_mean['Model'],
                model_feature_names,
                max_lf,
                f'Model mean |SHAP| (fold mean, {len(averages)} folds)',
                temporal_mean,
                feature_series,
                diverging=False,
                support=support,
                temporal_title='Temporal importance (mean ± std across folds)',
                feature_title='Feature importance (mean ± std across folds)',
                temporal_err=temporal_std,
                feature_err=feature_err)

        fig.suptitle(f'Fold-averaged |SHAP| vs ground truth — {scenario}',
                     fontweight='bold',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figs.append(fig)
    return tuple(figs) if figs else None
