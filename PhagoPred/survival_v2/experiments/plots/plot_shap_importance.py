from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import warnings
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.plots import (
    SERIES,
    DatasetAverage,
    heatmap_panel,
    temporal_panel,
    feature_panel,
    missing_panel,
    dataset_average,
    sample_indices,
    read_samples,
    sample_data,
    present_series,
    observed_length,
    event_str,
    label as series_label,
)
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()

_SHAP_FILE = 'shap.h5'  # written by interpret.run_shap


def _n_cols(keys: list[str]) -> int:
    return len(keys) + 2  # one heatmap per series, then temporal, then feature


def _average_keys(averages: list[DatasetAverage]) -> list[str]:
    """Series present in at least one average, in display order."""
    return [
        k for k in SERIES if any(a.temporal.get(k) is not None
                                 for a in averages)
    ]


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
    """NaN-pad the last axis on the left out to ``length`` (no-op if already
    long enough), keeping arrays right-aligned on the landmark."""
    pad_width = length - arr.shape[-1]
    if pad_width <= 0:
        return arr
    widths = [(0, 0)] * (arr.ndim - 1) + [(pad_width, 0)]
    return np.pad(arr, widths, constant_values=np.nan)


def _shap_path(record: ExperimentRecord) -> Path | None:
    if record.experiment_dir is None:
        return None
    path = Path(record.experiment_dir) / _SHAP_FILE
    if path.is_file():
        return path
    log.warning(f'No {_SHAP_FILE} found in {record.experiment_dir}')
    return None


def _common_indices(paths: list[Path]) -> list[int]:
    """Sample indices present in every file, so experiments on one dataset
    are compared on the same samples (files can hold different numbers, e.g.
    after the sample count was lowered)."""
    common = set(sample_indices(paths[0]))
    for path in paths[1:]:
        common &= set(sample_indices(path))
    return sorted(common)


def _row(
    axes,
    keys: list[str],
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
    """One row of axes: one heatmap per series in ``keys``, then temporal,
    then feature — the layout ``interpret.plots.plot_dataset_average`` uses
    for a single experiment, repeated as a row per varying-param combination
    so experiments can be compared directly."""
    for i, key in enumerate(keys):
        ax = axes[i]
        matrix = heatmaps.get(key)
        if matrix is None:
            missing_panel(ax, series_label(key))
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

        common = _common_indices([p for _, p in usable])
        averages = [
            dataset_average(p, normalise=normalise, indices=common)
            for _, p in usable
        ]
        keys = _average_keys(averages)
        unit = 'relative' if normalise else 'mean |SHAP|'
        fig, axes = plt.subplots(len(usable),
                                 _n_cols(keys),
                                 figsize=(6 * _n_cols(keys), 3.6 * len(usable)),
                                 squeeze=False)
        for row, ((record, _), avg) in enumerate(zip(usable, averages)):
            _row(axes[row],
                 keys,
                 _model_label(record, varying_params),
                 avg.heatmaps,
                 avg.feature_names,
                 avg.n_frames,
                 lambda key, unit=unit: f'{series_label(key)}  {unit}',
                 {k: avg.temporal.get(k) for k in keys},
                 {
                     k: (avg.feature_names, avg.feature[k])
                     for k in keys if avg.feature.get(k) is not None
                 },
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
    varying-param combination (repeats or kfold splits of the same model)
    are collapsed into one row: temporal and feature curves show mean +/- std
    across folds, heatmaps the fold mean only.

    Ground-truth series are identical across folds when the folds share a
    SHAP shard (their std is then 0), so this mainly shows model spread.
    """
    figs = []
    for scenario, records in _group_by_dataset(experiments).items():
        groups: defaultdict[tuple, list[ExperimentRecord]] = defaultdict(list)
        for record in records:
            groups[_model_key(record, varying_params)].append(record)

        paths = [p for r in records if (p := _shap_path(r)) is not None]
        common = _common_indices(paths) if paths else []
        group_averages = {
            key: [
                dataset_average(p, normalise=normalise, indices=common)
                for r in recs if (p := _shap_path(r)) is not None
            ] for key, recs in groups.items()
        }
        keys = _average_keys(
            [a for avgs in group_averages.values() for a in avgs])
        if not keys:
            log.warning(f'{scenario}: no SHAP files; skipping fold average')
            continue

        unit = 'relative' if normalise else 'mean |SHAP|'
        fig, axes = plt.subplots(len(groups),
                                 _n_cols(keys),
                                 figsize=(6 * _n_cols(keys), 3.6 * len(groups)),
                                 squeeze=False)
        for row, (key, averages) in enumerate(group_averages.items()):
            label = _group_label(key, varying_params)
            if not averages:
                missing_panel(axes[row, 0], 'SHAP')
                axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')
                for ax in axes[row, 1:]:
                    ax.set_axis_off()
                continue

            n_frames = max(a.n_frames for a in averages)
            feature_names = averages[0].feature_names

            heat_mean: dict[str, np.ndarray | None] = {}
            temporal_mean: dict[str, np.ndarray | None] = {}
            temporal_std: dict[str, np.ndarray] = {}
            feature_series: dict[str, tuple] = {}
            feature_err: dict[str, tuple] = {}
            for k in keys:
                heat_arrs = [
                    _pad(a.heatmaps[k], n_frames) for a in averages
                    if a.heatmaps.get(k) is not None
                ]
                heat_mean[k] = _nanmean(
                    np.stack(heat_arrs)) if heat_arrs else None

                temp_arrs = [
                    _pad(a.temporal[k], n_frames) for a in averages
                    if a.temporal.get(k) is not None
                ]
                if temp_arrs:
                    temp_stack = np.stack(temp_arrs)
                    temporal_mean[k] = _nanmean(temp_stack)
                    temporal_std[k] = _nanstd(temp_stack)
                else:
                    temporal_mean[k] = None

                feat_arrs = [
                    a.feature[k] for a in averages
                    if a.feature.get(k) is not None
                ]
                if feat_arrs:
                    feat_stack = np.stack(feat_arrs)
                    feature_series[k] = (feature_names, _nanmean(feat_stack))
                    feature_err[k] = (feature_names, _nanstd(feat_stack))

            support = np.nansum(np.stack(
                [_pad(a.support, n_frames) for a in averages]),
                                axis=0)

            _row(axes[row],
                 keys,
                 label,
                 heat_mean,
                 feature_names,
                 n_frames,
                 lambda key, unit=unit, n=len(averages):
                 (f'{series_label(key)}  fold mean ({n} folds)  {unit}'),
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
    dataset. Signed SHAP (diverging colour scale), +/- one standard error on
    the temporal and feature panels. Only samples that are the same sample
    (same seed) in every experiment's file are plotted."""
    figs = []
    for scenario, records in _group_by_dataset(experiments).items():
        usable = [(r, p) for r in records if (p := _shap_path(r)) is not None]
        if not usable:
            log.warning(f'{scenario}: no experiments with a SHAP file; '
                        'skipping SHAP sample plots')
            continue

        candidates = _common_indices([p for _, p in usable])[:num_plot_samples]
        if not candidates:
            log.warning(f'{scenario}: no sample indices common to all '
                        'experiments')
            continue
        per_file = [read_samples(p, candidates) for _, p in usable]

        for j, sample_idx in enumerate(candidates):
            row_samples = [samples[j] for samples in per_file]
            if len({s.seed for s in row_samples}) > 1:
                log.warning(f'{scenario}: sample {sample_idx} differs across '
                            'experiments; skipping')
                continue
            keys = present_series(row_samples)
            fig, axes = plt.subplots(len(usable),
                                     _n_cols(keys),
                                     figsize=(6 * _n_cols(keys),
                                              3.6 * len(usable)),
                                     squeeze=False)
            for row, ((record, _), sample) in enumerate(zip(usable,
                                                            row_samples)):
                d = sample_data(sample, keys, normalise)
                names = sample.feature_names
                _row(axes[row],
                     keys,
                     _model_label(record, varying_params),
                     d.heatmaps,
                     names,
                     observed_length(sample),
                     lambda key: f'{series_label(key)} SHAP',
                     d.temporal,
                     {
                         k: (names, v)
                         for k, v in d.feature.items() if v is not None
                     },
                     diverging=True,
                     temporal_err=d.temporal_err,
                     feature_err={
                         k: (names, v)
                         for k, v in d.feature_err.items()
                     })

            sample = row_samples[0]
            fig.suptitle(
                f'SHAP vs config — {scenario}, sample {sample_idx} '
                f'(L={sample.landmark_frame}, event={event_str(sample)})',
                fontweight='bold',
                fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            figs.append(fig)
    return tuple(figs) if figs else None
