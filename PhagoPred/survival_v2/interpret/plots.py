"""Plotting SHAP files written by ``run_shap`` (``<experiment>/shap.h5``, or a
shared GT samples file).

Every (predictor, background) result is a *series*, keyed
``'<predictor>_<background>'`` (e.g. ``'ground_truth_interventional'``,
``'model_observational_var'``). Colour encodes the background, line style /
hatching the predictor, so GT and model under the same background sit next to
each other.

Time axes are aligned on the landmark (0 = landmark, negative = before it).
SHAP over time is per segment; ``per_frame`` spreads each segment's value
evenly over its frames so samples of different lengths can be averaged.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from .data_models import (BackgroundEnum, PredictorEnum, SampleWithSHAP,
                          SHAPResult)
from .utils.io import read_sample
from .utils.io import sample_indices as _group_sample_indices

log = get_logger()

LANDMARK_XLABEL = 'frames relative to landmark'

_BG_COLOUR = {
    BackgroundEnum.INTERVENTIONAL: 'tab:orange',
    BackgroundEnum.OBSERVATIONAL_VAR: 'tab:green',
    BackgroundEnum.OBSERVATIONAL_GT: 'tab:purple',
}
_PRED_LINE = {PredictorEnum.GROUND_TRUTH: '-', PredictorEnum.MODEL: '--'}
_PRED_HATCH = {PredictorEnum.GROUND_TRUTH: None, PredictorEnum.MODEL: '//'}

# Series in display order: per background, GT then model. The model is never
# explained under OBSERVATIONAL_GT (its fills are clean), so that pair is left
# out.
SERIES: dict[str, tuple[PredictorEnum, BackgroundEnum]] = {
    f'{p.value}_{b.value}': (p, b)
    for b in BackgroundEnum for p in PredictorEnum
    if not (p == PredictorEnum.MODEL and b == BackgroundEnum.OBSERVATIONAL_GT)
}
# Kept for callers of the old module.
SHAP_PANELS = set(SERIES)
PANEL_ORDER = ('signals', *SERIES, 'temporal', 'feature')


def series_style(key: str) -> dict:
    p, b = SERIES[key]
    return dict(color=_BG_COLOUR[b], ls=_PRED_LINE[p], lw=1.2)


def label(key: str) -> str:
    p, b = SERIES[key]
    pred = 'GT' if p == PredictorEnum.GROUND_TRUTH else 'Model'
    return f"{pred} {b.value.replace('_', ' ')}"


# === reading ===


def sample_indices(h5_path: Path | str) -> list[int]:
    with h5py.File(h5_path, 'r') as f:
        return _group_sample_indices(f)


def read_samples(h5_path: Path | str,
                 indices: list[int] | None = None) -> list[SampleWithSHAP]:
    with h5py.File(h5_path, 'r') as f:
        idxs = _group_sample_indices(f) if indices is None else indices
        return [read_sample(f, i) for i in idxs]


def get_result(sample: SampleWithSHAP, key: str) -> SHAPResult | None:
    return sample.shap_vals.get(SERIES[key])


def present_series(samples: list[SampleWithSHAP]) -> list[str]:
    """Series that at least one sample has, in display order."""
    have = {k for s in samples for k in s.shap_vals}
    return [k for k, pb in SERIES.items() if pb in have]


def observed_length(sample: SampleWithSHAP) -> int:
    """Frames before the landmark that were explained."""
    return min(sample.landmark_frame, sample.feature_vals.features.shape[1])


def event_str(sample: SampleWithSHAP) -> str:
    return ('censored'
            if sample.event_frame is None else f'{sample.event_frame:.0f}')


# === per-frame values ===


def spread_segments(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Per-segment values (segments on the last axis) -> per-frame density,
    each segment's value divided evenly over the frames it spans."""
    seg_len = np.diff(boundaries)
    return np.repeat(values / seg_len, seg_len, axis=-1)


def per_frame(values: np.ndarray | None,
              boundaries: np.ndarray | None) -> np.ndarray | None:
    if values is None or boundaries is None:
        return values
    if values.shape[-1] == len(boundaries) - 1:
        return spread_segments(values, boundaries)
    return values


def _normaliser(values: np.ndarray | None, normalise: bool) -> float:
    if values is None or not normalise:
        return 1.0
    total = float(np.nansum(np.abs(values)))
    return total if total > 0 else 1.0


# === panels ===


def heatmap_panel(ax: plt.Axes,
                  matrix: np.ndarray,
                  row_labels: list[str],
                  n_frames: int,
                  title: str,
                  row_normalise: bool = False,
                  diverging: bool = True) -> None:
    """(feature x frame) heatmap; the right edge is the landmark."""
    m = np.asarray(matrix, dtype=float)
    if row_normalise:
        scale = np.nanmax(np.abs(m), axis=1, keepdims=True)
        m = m / np.where(scale == 0.0, 1.0, scale)
    vmax = max(float(np.nanmax(np.abs(m))), 1e-9)
    im = ax.imshow(m,
                   aspect='auto',
                   origin='lower',
                   cmap='RdBu_r' if diverging else 'Greys',
                   vmin=-vmax if diverging else 0.0,
                   vmax=vmax,
                   extent=[-n_frames, 0, -0.5, m.shape[0] - 0.5],
                   interpolation='nearest')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel(LANDMARK_XLABEL, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def temporal_panel(ax: plt.Axes,
                   series: dict[str, np.ndarray | None],
                   title: str,
                   n_frames: int | None = None,
                   support: np.ndarray | None = None,
                   err: dict[str, np.ndarray] | None = None) -> None:
    """Per-frame importance, one line per series, right-aligned on the
    landmark. ``err`` shades +/- a band; ``support`` (averages) shades how
    many samples reach each frame."""
    for key, values in series.items():
        if values is None:
            continue
        style = series_style(key)
        x = np.arange(-len(values), 0)
        ax.plot(x, values, label=label(key), **style)
        spread = None if err is None else err.get(key)
        if spread is not None:
            ax.fill_between(x,
                            values - spread,
                            values + spread,
                            color=style['color'],
                            alpha=0.15,
                            linewidth=0)
    if support is not None:
        twin = ax.twinx()
        twin.fill_between(np.arange(-len(support), 0),
                          support,
                          color='grey',
                          alpha=0.12,
                          lw=0)
        twin.set_ylabel('samples contributing', fontsize=7, color='grey')
        twin.tick_params(axis='y', labelsize=7, colors='grey')
        twin.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
    if n_frames is not None:
        ax.set_xlim(-n_frames, 0)
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_xlabel(LANDMARK_XLABEL, fontsize=8)
    ax.set_ylabel('importance', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)


def feature_panel(ax: plt.Axes,
                  series: dict[str, tuple[list[str], np.ndarray]],
                  title: str,
                  err: dict[str, tuple[list[str], np.ndarray]] | None = None
                  ) -> None:
    """Grouped per-feature bars, aligned by feature name across series."""
    names: list[str] = []
    for feature_names, _ in series.values():
        names += [n for n in feature_names if n not in names]
    positions = np.arange(len(names))
    width = 0.8 / max(len(series), 1)

    for i, (key, (feature_names, values)) in enumerate(series.items()):
        lookup = dict(zip(feature_names, values))
        yerr = None
        if err is not None and key in err:
            err_lookup = dict(zip(*err[key]))
            yerr = [err_lookup.get(n, 0.0) for n in names]
        p, b = SERIES[key]
        ax.bar(positions + (i - (len(series) - 1) / 2) * width,
               [lookup.get(n, np.nan) for n in names],
               yerr=yerr,
               capsize=2,
               width=width,
               label=label(key),
               color=_BG_COLOUR[b],
               hatch=_PRED_HATCH[p],
               edgecolor='white' if _PRED_HATCH[p] else None,
               alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=7, rotation=90, ha='center')
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylabel('importance', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)


def missing_panel(ax: plt.Axes, what: str) -> None:
    ax.text(0.5,
            0.5,
            f'no {what}',
            ha='center',
            va='center',
            fontsize=8,
            transform=ax.transAxes)
    ax.set_axis_off()


# === one sample ===


@dataclass
class SampleData:
    """Signed per-series arrays for one sample, per-frame, optionally scaled
    so each series' |values| sum to 1 (stderrs scaled the same)."""
    heatmaps: dict[str, np.ndarray | None]  # (F, T)
    temporal: dict[str, np.ndarray | None]  # (T,)
    temporal_err: dict[str, np.ndarray]
    feature: dict[str, np.ndarray | None]  # (F,)
    feature_err: dict[str, np.ndarray]


def sample_data(sample: SampleWithSHAP,
                keys: list[str],
                normalise: bool = True) -> SampleData:
    out = SampleData({}, {}, {}, {}, {})
    for key in keys:
        r = get_result(sample, key)
        if r is None:
            out.heatmaps[key] = out.temporal[key] = out.feature[key] = None
            continue
        b = r.segment_boundaries
        out.heatmaps[key] = per_frame(r.temporal_feature, b)
        if r.temporal is not None:
            z = _normaliser(r.temporal, normalise)
            out.temporal[key] = per_frame(r.temporal, b) / z
            if r.temporal_stderr is not None:
                out.temporal_err[key] = per_frame(r.temporal_stderr, b) / z
        else:
            out.temporal[key] = None
        if r.feature is not None:
            z = _normaliser(r.feature, normalise)
            out.feature[key] = r.feature / z
            if r.feature_stderr is not None:
                out.feature_err[key] = r.feature_stderr / z
        else:
            out.feature[key] = None
    return out


def plot_sample(sample: SampleWithSHAP,
                keys: list[str] | None = None,
                title: str | None = None,
                normalise: bool = True) -> plt.Figure:
    """Signals, one signed heatmap per series, then temporal and feature
    panels (+/- one standard error)."""
    keys = keys or present_series([sample])
    n_obs = observed_length(sample)
    fig, axes = plt.subplots(1,
                             len(keys) + 3,
                             figsize=(5.2 * (len(keys) + 3), 4),
                             squeeze=False)
    axes = list(axes[0])
    heatmap_panel(axes[0],
                  sample.feature_vals.features[:, :n_obs],
                  sample.feature_names,
                  n_obs,
                  f'Signals (row-normalised)  L={sample.landmark_frame}  '
                  f'event={event_str(sample)}',
                  row_normalise=True)
    d = sample_data(sample, keys, normalise)
    for ax, key in zip(axes[1:], keys):
        if d.heatmaps[key] is None:
            missing_panel(ax, label(key))
        else:
            heatmap_panel(ax, d.heatmaps[key], sample.feature_names, n_obs,
                          f'{label(key)} SHAP')
    unit = ' (relative)' if normalise else ''
    temporal_panel(axes[-2],
                   d.temporal,
                   f'Temporal{unit}',
                   n_frames=n_obs,
                   err=d.temporal_err)
    feature_panel(axes[-1], {
        k: (sample.feature_names, v)
        for k, v in d.feature.items() if v is not None
    },
                  f'Feature{unit}',
                  err={
                      k: (sample.feature_names, v)
                      for k, v in d.feature_err.items()
                  })
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_samples(h5_path: Path | str,
                 save_dir: Path | str,
                 num_plot_samples: int = 5,
                 title_prefix: str = '') -> list[Path]:
    """One figure per sample (the first ``num_plot_samples``)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    idxs = sample_indices(h5_path)[:num_plot_samples]
    saved = []
    for i, sample in zip(idxs, read_samples(h5_path, idxs)):
        name = f'sample {i}'
        fig = plot_sample(
            sample,
            title=f'{title_prefix} — {name}' if title_prefix else name)
        path = save_dir / f'sample_{i:02d}.png'
        fig.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        saved.append(path)
    log.info(f'Saved {len(saved)} sample figures to {save_dir}')
    return saved


# === dataset average ===


@dataclass
class DatasetAverage:
    """Per-series mean |SHAP| over every sample in a file, right-aligned on
    the landmark (column -k is k frames before it)."""
    feature_names: list[str]
    n_frames: int  # longest window
    num_samples: int
    heatmaps: dict[str, np.ndarray | None]  # (F, n_frames)
    temporal: dict[str, np.ndarray | None]  # (n_frames,)
    feature: dict[str, np.ndarray | None]  # (F,)
    support: np.ndarray  # (n_frames,) samples reaching back to each frame


def dataset_average(h5_path: Path | str,
                    normalise: bool = True,
                    keys: list[str] | None = None,
                    indices: list[int] | None = None) -> DatasetAverage:
    """Mean |SHAP| per series. Absolute values are taken *before* averaging,
    so a feature that sometimes raises and sometimes lowers the prediction
    doesn't cancel to zero. ``normalise`` rescales each averaged series to
    sum to 1, comparing estimators on relative importance (GT and model live
    on different output scales). ``indices``: only these samples (default
    all)."""
    samples = read_samples(h5_path, indices)
    keys = keys or list(SERIES)
    feature_names = samples[0].feature_names
    n_frames = max(observed_length(s) for s in samples)
    F = len(feature_names)

    support = np.zeros(n_frames)
    for s in samples:
        support[n_frames - observed_length(s):] += 1

    heatmaps, temporal, feature = {}, {}, {}
    for key in keys:
        heat = np.full((len(samples), F, n_frames), np.nan)
        temp = np.full((len(samples), n_frames), np.nan)
        feat = np.full((len(samples), F), np.nan)
        for i, s in enumerate(samples):
            r = get_result(s, key)
            if r is None:
                continue
            b = r.segment_boundaries
            if r.temporal_feature is not None:
                tf = np.abs(per_frame(r.temporal_feature, b))
                heat[i, :, n_frames - tf.shape[1]:] = tf
            if r.temporal is not None:
                t = np.abs(per_frame(r.temporal, b))
                temp[i, n_frames - len(t):] = t
            if r.feature is not None:
                feat[i] = np.abs(r.feature)
        heatmaps[key] = _mean_or_none(heat, normalise)
        temporal[key] = _mean_or_none(temp, normalise)
        feature[key] = _mean_or_none(feat, normalise)
    return DatasetAverage(feature_names, n_frames, len(samples), heatmaps,
                          temporal, feature, support)


def _mean_or_none(stack: np.ndarray, normalise: bool) -> np.ndarray | None:
    if np.all(np.isnan(stack)):
        return None
    with warnings.catch_warnings():  # all-NaN columns (short windows)
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    return mean / _normaliser(mean, normalise)


def plot_dataset_average(h5_path: Path | str,
                         title: str | None = None,
                         normalise: bool = True) -> plt.Figure:
    avg = dataset_average(h5_path, normalise=normalise)
    keys = [k for k in SERIES if avg.temporal.get(k) is not None]
    unit = 'relative' if normalise else 'mean |SHAP|'
    fig, axes = plt.subplots(1,
                             len(keys) + 2,
                             figsize=(6 * (len(keys) + 2), 4),
                             squeeze=False)
    axes = list(axes[0])
    for ax, key in zip(axes, keys):
        heatmap_panel(ax,
                      avg.heatmaps[key],
                      avg.feature_names,
                      avg.n_frames,
                      f'{label(key)}  {unit}',
                      diverging=False)
    temporal_panel(axes[-2], {k: avg.temporal[k]
                              for k in keys},
                   f'Temporal importance  {unit}',
                   n_frames=avg.n_frames,
                   support=avg.support)
    feature_panel(axes[-1], {k: (avg.feature_names, avg.feature[k])
                             for k in keys}, f'Feature importance  {unit}')
    suptitle = f'Dataset average over {avg.num_samples} samples'
    if title:
        suptitle += f' — {title}'
    fig.suptitle(suptitle, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
