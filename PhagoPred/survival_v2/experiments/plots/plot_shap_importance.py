from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.data_models import SampleWithSHAP
from PhagoPred.survival_v2.interpret.plots import (
    PANEL_ORDER,
    SHAP_PANELS,
    heatmap_panel,
    temporal_panel,
    feature_panel,
    dataset_average,
    sample_indices,
    spread_segments,
)
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()

_SHAP_FILE = 'shap_samples.h5'
_KEYS = [k for k in PANEL_ORDER if k in SHAP_PANELS]
_N_COLS = len(_KEYS) + 2  # one heatmap per SHAP key, then temporal, then feature


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

    Unlike ``_model_label`` (which falls back to the experiment directory
    name, unique per run), this is constant across repeats of a config so
    they can be averaged together.
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


def _pad(arr: np.ndarray, length: int) -> np.ndarray:
    """NaN-pad the last axis out to ``length`` (no-op if already long enough)."""
    pad_width = length - arr.shape[-1]
    if pad_width <= 0:
        return arr
    widths = [(0, 0)] * (arr.ndim - 1) + [(0, pad_width)]
    return np.pad(arr, widths, constant_values=np.nan)


def _shap_path(record: ExperimentRecord) -> Path | None:
    if record.experiment_dir is None:
        return None
    path = Path(record.experiment_dir) / _SHAP_FILE
    if path.is_file():
        return path
    log.warning(f'No {_SHAP_FILE} found in {record.experiment_dir}')
    return None


def _row(
    axes,
    label: str,
    heatmaps: dict[str, np.ndarray | None],
    feature_names: list[str],
    n_frames: int,
    heat_title: Callable[[str], str],
    temporal_series: dict,
    feature_series: dict,
    diverging: bool,
    support: np.ndarray | None = None,
    temporal_title: str = 'Temporal importance',
    feature_title: str = 'Feature importance',
    temporal_err: dict | None = None,
    feature_err: dict | None = None,
) -> None:
    """One row of ``_N_COLS`` axes: one heatmap per SHAP key, then temporal,
    then feature — the same column layout ``interpret.plots.plot_dataset_average``
    uses for a single experiment, repeated as a row per varying-param
    combination so experiments can be compared directly."""
    for i, key in enumerate(_KEYS):
        ax = axes[i]
        matrix = heatmaps.get(key)
        if matrix is None:
            ax.text(0.5,
                    0.5,
                    f'no {key}',
                    ha='center',
                    va='center',
                    fontsize=8,
                    transform=ax.transAxes)
            ax.set_axis_off()
            continue
        heatmap_panel(ax,
                      matrix,
                      feature_names,
                      n_frames,
                      heat_title(key),
                      diverging=diverging)
    temporal_panel(axes[-2],
                   temporal_series,
                   temporal_title,
                   n_frames=n_frames,
                   support=support,
                   err=temporal_err)
    feature_panel(axes[-1], feature_series, feature_title, err=feature_err)
    axes[0].set_ylabel(label, fontsize=9, fontweight='bold')


def _get_shap_result(sample: SampleWithSHAP, key: str):
    explainer = [e for e in sample.explainers() if e in key]
    background = [b for b in sample.backgrounds() if b in key]
    if len(explainer) == 0 or len(background) == 0:
        return None
    return sample.shap_vals[(explainer[0], background[0])]


def _sample_data(
    sample: SampleWithSHAP, normalise: bool
) -> tuple[dict[str, np.ndarray | None], dict[str, np.ndarray | None],
          dict[str, np.ndarray | None]]:
    """Signed, per-key heatmap/temporal/feature arrays for one sample (no
    averaging) — the row-comparison counterpart of
    ``interpret.plots.plot_sample_on_axes``."""
    lf = sample.landmark_frame

    def _norm(values):
        values = np.asarray(values, dtype=float)
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    heatmaps: dict[str, np.ndarray | None] = {}
    temporal: dict[str, np.ndarray | None] = {}
    feature: dict[str, np.ndarray | None] = {}
    for key in _KEYS:
        result = _get_shap_result(sample, key)
        if result is None:
            heatmaps[key] = temporal[key] = feature[key] = None
            continue

        if result.temporal_feature is not None:
            tf = result.temporal_feature
            if result.segment_boundaries is not None and tf.shape[-1] != lf:
                tf = spread_segments(tf, result.segment_boundaries)
            heatmaps[key] = tf
        else:
            heatmaps[key] = None

        if result.temporal is not None:
            values = result.temporal
            if result.segment_boundaries is not None and len(values) != lf:
                values = spread_segments(values, result.segment_boundaries)
            temporal[key] = _norm(values)
        else:
            temporal[key] = None

        feature[key] = None if result.feature is None else _norm(
            result.feature)

    return heatmaps, temporal, feature


def _group_by_dataset(
    experiments: list[ExperimentRecord]
) -> defaultdict[str, list[ExperimentRecord]]:
    grouped: defaultdict[str, list[ExperimentRecord]] = defaultdict(list)
    for exp in experiments:
        grouped[exp.experiemnt_cfg.dataset.name].append(exp)
    return grouped


def plot_shap_average_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per dataset: dataset-average |SHAP|, one row per experiment."""
    figs = []
    for scenario, records in _group_by_dataset(experiments).items():
        usable = [(r, p) for r in records if (p := _shap_path(r)) is not None]
        if not usable:
            log.warning(f'{scenario}: no experiments with a SHAP file; '
                        'skipping SHAP average plot')
            continue

        unit = 'relative' if normalise else 'mean |SHAP|'
        fig, axes = plt.subplots(len(usable),
                                 _N_COLS,
                                 figsize=(6 * _N_COLS, 3.6 * len(usable)),
                                 squeeze=False)
        for row, (record, path) in enumerate(usable):
            avg = dataset_average(path, normalise=normalise)
            temporal_series = dict(avg.temporal)
            feature_series = {
                k: (avg.feature_names, v)
                for k, v in avg.feature.items() if v is not None
            }
            _row(axes[row],
                 _model_label(record, varying_params),
                 avg.heatmaps,
                 avg.feature_names,
                 avg.max_lf,
                 lambda key, unit=unit: f"{key.replace('_', ' ').capitalize()}  {unit}",
                 temporal_series,
                 feature_series,
                 diverging=False,
                 support=avg.support,
                 temporal_title=f'Temporal importance  {unit}',
                 feature_title=f'Feature importance  {unit}')

        fig.suptitle(f'Dataset-average |SHAP| — {scenario}',
                     fontweight='bold',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figs.append(fig)
    return tuple(figs) if figs else None


def plot_shap_fold_average_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per dataset: mean +/- std |SHAP| across folds/repeats, one
    row per varying-param combination.

    Companion to ``plot_shap_average_across_models``, which draws each
    repeat's dataset-average as its own row. Here, records sharing the same
    varying-param combination (i.e. repeats of the same model, or kfold
    splits of it) are collapsed into a single row: temporal and feature
    curves show mean +/- std across folds, and each heatmap shows the fold
    mean only (a spread doesn't render on a heatmap).
    """
    figs = []
    for scenario, records in _group_by_dataset(experiments).items():
        groups: defaultdict[tuple, list[ExperimentRecord]] = defaultdict(list)
        for record in records:
            groups[_model_key(record, varying_params)].append(record)

        unit = 'relative' if normalise else 'mean |SHAP|'
        fig, axes = plt.subplots(len(groups),
                                 _N_COLS,
                                 figsize=(6 * _N_COLS, 3.6 * len(groups)),
                                 squeeze=False)
        for row, (key, group_records) in enumerate(groups.items()):
            label = _group_label(key, varying_params)
            paths = [p for r in group_records if (p := _shap_path(r)) is not None]
            averages = [dataset_average(p, normalise=normalise) for p in paths]
            if not averages:
                axes[row, 0].text(0.5,
                                  0.5,
                                  'no SHAP',
                                  ha='center',
                                  va='center',
                                  transform=axes[row, 0].transAxes)
                axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')
                for ax in axes[row]:
                    ax.set_axis_off()
                continue

            max_lf = max(a.max_lf for a in averages)
            feature_names = averages[0].feature_names

            heat_mean: dict[str, np.ndarray | None] = {}
            temporal_mean: dict[str, np.ndarray | None] = {}
            temporal_std: dict[str, np.ndarray] = {}
            feature_series: dict[str, tuple] = {}
            feature_err: dict[str, tuple] = {}
            for k in _KEYS:
                heat_arrs = [
                    _pad(a.heatmaps[k], max_lf) for a in averages
                    if a.heatmaps.get(k) is not None
                ]
                heat_mean[k] = _nanmean(np.stack(heat_arrs)) if heat_arrs else None

                temp_arrs = [
                    _pad(a.temporal[k], max_lf) for a in averages
                    if a.temporal.get(k) is not None
                ]
                if temp_arrs:
                    temp_stack = np.stack(temp_arrs)
                    temporal_mean[k] = _nanmean(temp_stack)
                    temporal_std[k] = _nanstd(temp_stack)
                else:
                    temporal_mean[k] = None

                feat_arrs = [
                    a.feature[k] for a in averages if a.feature.get(k) is not None
                ]
                if feat_arrs:
                    feat_stack = np.stack(feat_arrs)
                    feature_series[k] = (feature_names, _nanmean(feat_stack))
                    feature_err[k] = (feature_names, _nanstd(feat_stack))

            support = np.nansum(np.stack(
                [_pad(a.support, max_lf) for a in averages]),
                                axis=0)

            _row(
                axes[row],
                label,
                heat_mean,
                feature_names,
                max_lf,
                lambda key, unit=unit, n=len(averages): (
                    f"{key.replace('_', ' ').capitalize()}  fold mean "
                    f"({n} folds)  {unit}"),
                temporal_mean,
                feature_series,
                diverging=False,
                support=support,
                temporal_title='Temporal importance (mean ± std across folds)',
                feature_title='Feature importance (mean ± std across folds)',
                temporal_err=temporal_std,
                feature_err=feature_err)

        fig.suptitle(f'Fold-averaged |SHAP| — {scenario}',
                     fontweight='bold',
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        figs.append(fig)
    return tuple(figs) if figs else None


def plot_shap_samples_across_models(
        experiments: list[ExperimentRecord],
        varying_params: dict,
        num_plot_samples: int = 5,
        normalise: bool = True) -> tuple[plt.Figure, ...] | None:
    """One figure per (dataset, sample), rows = experiments sharing that
    dataset. Signed SHAP, diverging colour scale, only samples present in
    every experiment's file are plotted."""
    figs = []
    for scenario, records in _group_by_dataset(experiments).items():
        usable = [(r, p) for r in records if (p := _shap_path(r)) is not None]
        if not usable:
            log.warning(f'{scenario}: no experiments with a SHAP file; '
                        'skipping SHAP sample plots')
            continue

        common = set(sample_indices(usable[0][1]))
        for _, path in usable[1:]:
            common &= set(sample_indices(path))
        indices = sorted(common)[:num_plot_samples]
        if not indices:
            log.warning(f'{scenario}: no sample indices common to all '
                        'experiments')
            continue

        for sample_idx in indices:
            fig, axes = plt.subplots(len(usable),
                                     _N_COLS,
                                     figsize=(6 * _N_COLS, 3.6 * len(usable)),
                                     squeeze=False)
            lf = death_str = None
            for row, (record, path) in enumerate(usable):
                sample = SampleWithSHAP.read_h5(path, sample_idx)
                lf = sample.landmark_frame
                death_str = ('censored' if sample.death_frame is None else
                             f'{sample.death_frame:.0f}')
                heatmaps, temporal_series, feature_vals = _sample_data(
                    sample, normalise)
                feature_series = {
                    k: (sample.feature_names, v)
                    for k, v in feature_vals.items() if v is not None
                }
                _row(axes[row],
                     _model_label(record, varying_params),
                     heatmaps,
                     sample.feature_names,
                     lf,
                     lambda key: f"{key.replace('_', ' ').capitalize()} SHAP",
                     temporal_series,
                     feature_series,
                     diverging=True)

            fig.suptitle(
                f'SHAP vs config — {scenario}, sample {sample_idx} '
                f'(lf={lf}, death={death_str})',
                fontweight='bold',
                fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            figs.append(fig)
    return tuple(figs) if figs else None
