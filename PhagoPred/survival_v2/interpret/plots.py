from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.data_models import SampleWithSHAP, SHAPResult
# from PhagoPred.survival_v2.interpret_old.importance_data import (
#     dataset_average,
#     horizon_outputs,
#     load_sample_importances,
#     read_root_attrs,
#     sample_indices,
# )

log = get_logger()

# Series whose feature axis is the model's, not the graph's (no Hazard node).
# MODEL_KEYS = ('Model', )

SERIES_STYLE = {
    'ground_truth_interventional': dict(color='tab:orange', lw=1.2),
    'kernel_interventional': dict(color='tab:green', lw=1.2),
    'ground_truth_observational': dict(color='tab:purple', lw=1.2),
    'kernel_observational': dict(color='tab:pink', lw=1.2),
}
PANEL_ORDER = ('signals', 'ground_truth_interventional',
               'kernel_interventional', 'ground_truth_observational',
               'kernel_observational', 'temporal', 'feature', 'outputs')
SHAP_PANELS = {
    'ground_truth_interventional', 'kernel_interventional',
    'ground_truth_observational', 'kernel_observational'
}
DEFAULT_PANELS = {name: True for name in PANEL_ORDER}
# MODEL_ONLY_PANELS = {
#     'signals': True,
#     'interventional': False,
#     'observational': False,
#     'model': True,
#     'temporal': True,
#     'feature': True,
#     'outputs': False,
# }

# def series_rows(key: str, feature_names: list[str],
#                 model_feature_names: list[str] | None) -> list[str]:
#     """Feature axis a given series lives on."""
#     if key in MODEL_KEYS and model_feature_names is not None:
#         return model_feature_names
#     return feature_names


def heatmap_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    n_frames: int,
    title: str,
    row_normalise: bool = False,
    diverging: bool = True,
) -> None:
    """(feature x frame) heatmap on a shared [0, n_frames) axis.
    """
    m = np.asarray(matrix, dtype=float)
    if row_normalise:
        scale = np.nanmax(np.abs(m), axis=1, keepdims=True)
        m = m / np.where(scale == 0.0, 1.0, scale)
    extent = [0, n_frames, -0.5, m.shape[0] - 0.5]

    vmax = max(float(np.nanmax(np.abs(m))), 1e-9)
    vmin = -vmax if diverging else 0.0
    cmap = 'RdBu_r' if diverging else 'Greys'
    im = ax.imshow(m,
                   aspect='auto',
                   origin='lower',
                   cmap=cmap,
                   vmin=vmin,
                   vmax=vmax,
                   extent=extent,
                   interpolation='nearest')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('frame', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def spread_segments(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Per-segment values -> per-frame density, by dividing each segment's
    value evenly across the frames it spans. Works on a 1D ``(segments,)``
    array or a 2D ``(rows, segments)`` array (segments on the last axis).

    Ground-truth ``temporal``/``temporal_feature`` arrays are already
    per-frame; kernel SHAP's are per-segment (one value per player in the
    coarser temporal game). A single sample's per-segment values can be
    plotted as-is against a per-frame x axis — ``heatmap_panel``'s
    ``imshow(..., extent=...)`` stretches a lower-res image across the full
    frame range, and that's the only place ``plot_sample_on_axes`` needs this.
    But averaging *across* samples needs the actual per-frame values first:
    each sample's segments span different absolute frame ranges (segment
    boundaries scale with that sample's own landmark frame), so stacking
    per-segment columns directly would average unrelated frames together.
    """
    seg_len = np.diff(boundaries)
    density = values / seg_len
    return np.repeat(density, seg_len, axis=-1)


def temporal_panel(
    ax: plt.Axes,
    series: dict[str, np.ndarray],
    title: str,
    n_frames: int | None = None,
    support: np.ndarray | None = None,
    err: dict[str, np.ndarray] | None = None,
) -> None:
    """Per-frame importance curves, one line per estimator.

    Lines rather than bars: with landmark frames up to ~400, several overlaid bar
    series are unreadable. ``support`` (dataset averages only) shades how many
    samples reach each frame — the right-hand columns average over the handful of
    long-``lf`` samples and are correspondingly noisy. ``err`` (fold averages
    only) shades mean +/- err as a band per series, e.g. the std across
    kfold/repeat runs being averaged together.
    """
    for name, values in series.items():
        if values is None:
            continue
        color = SERIES_STYLE.get(name, {}).get('color')
        ax.plot(np.arange(len(values)),
                values,
                label=name,
                **SERIES_STYLE.get(name, {}))
        spread = None if err is None else err.get(name)
        if spread is not None:
            x = np.arange(len(values))
            ax.fill_between(x,
                            values - spread,
                            values + spread,
                            color=color,
                            alpha=0.2,
                            linewidth=0)
    if support is not None:
        twin = ax.twinx()
        twin.fill_between(np.arange(len(support)),
                          support,
                          color='grey',
                          alpha=0.12,
                          lw=0)
        twin.set_ylabel('samples contributing', fontsize=7, color='grey')
        twin.tick_params(axis='y', labelsize=7, colors='grey')
        twin.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
    if n_frames is not None:
        ax.set_xlim(0, n_frames)
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_xlabel('frame', fontsize=8)
    ax.set_ylabel('importance', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)


def outputs_panel(
    ax: plt.Axes,
    output_type: str,
    ground_truth: float | np.ndarray | None = None,
    model_prediction: np.ndarray | None = None,
    hazard_bins: np.ndarray | None = None,
    horizon: int | None = None,
    death_offset: float | None = None,
    title: str | None = None,
) -> None:
    """What the model predicted, next to what the graph actually implies.

    Unlike the SHAP panels this compares *outputs*, not attributions, and answers
    a different question: is the explanation even worth reading, i.e. did the
    model get this sample right? Pair ``ground_truth`` with
    ``importance_data.horizon_outputs``, which derives it from the stored hazard.

    Binary: two bars, P(event within horizon). Survival: the per-bin PMF, ground
    truth as bars and the model as a step over the bin edges. Either side may be
    None (an old file, or SHAP not yet run) and is simply left out.
    """
    if ground_truth is None and model_prediction is None:
        ax.text(0.5,
                0.5, 'No outputs stored.\nBackfill Horizon Hazard and rerun\n'
                'compare_importance.',
                ha='center',
                va='center',
                fontsize=8,
                transform=ax.transAxes)
        ax.set_axis_off()
        return

    model_colour = SERIES_STYLE['Model']['color']

    if output_type == 'binary':
        # A single scalar each: bars are more legible than a curve, and the
        # x axis is categorical, so no death marker belongs here.
        predicted = (np.nan if model_prediction is None else float(
            np.atleast_1d(model_prediction)[-1]))
        truth = np.nan if ground_truth is None else float(ground_truth)
        bars = ax.bar([0, 1], [truth, predicted],
                      0.5,
                      color=['w', 'k'],
                      edgecolor='k',
                      linewidth=1.0)
        for rect, value in zip(bars, (truth, predicted)):
            if not np.isfinite(value):
                continue
            # Probabilities near 1 would push the label outside ylim=(0, 1).
            inside = value > 0.92
            ax.text(rect.get_x() + rect.get_width() / 2,
                    value - 0.02 if inside else value + 0.02,
                    f'{value:.3f}',
                    ha='center',
                    va='top' if inside else 'bottom',
                    fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Ground truth', 'Model'], fontsize=8)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(
            f'P(event within {horizon} frames)' if horizon else 'P(event)',
            fontsize=8)
        ax.set_title(title or 'Predicted vs true event probability',
                     fontsize=8)
        return

    if hazard_bins is None:
        raise ValueError('survival output_type needs hazard_bins')
    edges = np.asarray(hazard_bins, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])

    if ground_truth is not None:
        ax.bar(centres,
               np.asarray(ground_truth, dtype=float),
               width=np.diff(edges) * 0.9,
               color='0.7',
               edgecolor='black',
               linewidth=0.6,
               label='Ground truth PMF')
    if model_prediction is not None:
        model_pmf = np.atleast_1d(model_prediction)
        if len(model_pmf) != len(centres):
            raise ValueError(f'model PMF has {len(model_pmf)} bins but '
                             f'hazard_bins defines {len(centres)}')
        # Step on the bin *edges* so the risers land on the bar boundaries even
        # when the bins are unequal width.
        ax.step(edges,
                np.append(model_pmf, model_pmf[-1]),
                where='post',
                color=model_colour,
                lw=1.5,
                label='Model PMF')
    if death_offset is not None and 0 <= death_offset <= edges[-1]:
        ax.axvline(death_offset,
                   color='red',
                   ls='--',
                   lw=1,
                   label=f'death @ lf+{death_offset:.0f}')

    ax.set_ylabel('probability mass', fontsize=8)
    ax.set_xlabel('frames past lf', fontsize=8)
    ax.set_title(title or 'Model output distribution vs ground truth PMF',
                 fontsize=8)
    ax.legend(fontsize=7)


def feature_panel(
    ax: plt.Axes,
    series: dict[str, tuple[list[str], np.ndarray]],
    title: str,
    err: dict[str, tuple[list[str], np.ndarray]] | None = None,
) -> None:
    """Grouped per-feature bars; series may cover different feature sets.

    Ground truth carries a Hazard bar the model has no counterpart for. It is
    left as a gap rather than dropped, so the feature axis stays aligned across
    series and across models. ``err`` (fold averages only) draws error bars,
    e.g. the std across kfold/repeat runs being averaged together.
    """
    names: list[str] = []
    for feature_names, _ in series.values():
        for name in feature_names:
            if name not in names:
                names.append(name)
    positions = np.arange(len(names))
    width = 0.8 / max(len(series), 1)

    for i, (label, (feature_names, values)) in enumerate(series.items()):
        lookup = dict(zip(feature_names, values))
        heights = [lookup.get(n, np.nan) for n in names]
        yerr = None
        if err is not None and label in err:
            err_names, err_values = err[label]
            err_lookup = dict(zip(err_names, err_values))
            yerr = [err_lookup.get(n, 0.0) for n in names]
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(positions + offset,
               heights,
               yerr=yerr,
               capsize=2,
               width=width,
               label=label,
               color=SERIES_STYLE.get(label, {}).get('color'),
               alpha=0.9)

    ax.set_xticks(positions)
    # Feature names run long; vertical labels keep them from overlapping.
    ax.set_xticklabels(names, fontsize=7, rotation=90, ha='center')
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylabel('importance', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)


def _resolve_panels(panels: dict | None) -> dict:
    merged = dict(DEFAULT_PANELS)
    if panels:
        merged.update(panels)
    return merged


def plot_sample_on_axes(
    h5_path: Path | str,
    axes: list[plt.Axes],
    sample_idx: int | None,
    panels: dict | None = None,
    normalise: bool = True,
) -> None:
    """Draw one stored sample across ``axes``, straight from an h5 SHAP file.
"""
    enabled = [panels[name] for name in panels]
    assert sum(enabled) == len(axes), (
        f'{sum(enabled)} panels enabled but {len(axes)} axes given')

    h5_path = Path(h5_path)

    sample = SampleWithSHAP.read_h5(h5_path, sample_idx)
    lf = sample.landmark_frame
    death_str = 'censored' if sample.death_frame is None else f'{sample.death_frame:.0f}'
    axes = list(axes)

    def _next() -> plt.Axes:
        return axes.pop(0)

    def _norm(values):
        values = np.asarray(values, dtype=float)
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    def _missing(ax: plt.Axes, what: str) -> None:
        ax.text(0.5,
                0.5,
                f'No {what} in\n{h5_path.name}',
                ha='center',
                va='center',
                fontsize=8,
                transform=ax.transAxes)
        ax.set_axis_off()

    def _get_shap_results(key: str) -> SHAPResult:
        explainer = [
            explainer for explainer in sample.explainers() if explainer in key
        ]
        background = [
            background for background in sample.backgrounds()
            if background in key
        ]
        if len(explainer) == 0 or len(background) == 0:
            return None
        explainer = explainer[0]
        background = background[0]
        return sample.shap_vals[(explainer, background)]

    # PLOT FEATURES
    if panels['signals']:
        ax = _next()
        if sample.feature_vals is None:
            _missing(ax, 'signals')
        else:
            heatmap_panel(
                ax,
                sample.feature_vals,
                sample.feature_names,
                lf,
                f'Signals (row-normalised)  lf={lf}  death={death_str}',
                row_normalise=True)

    # PLOT TEMPORAL_FEATURE HEATMAPS
    for key in [k for k in PANEL_ORDER if k in SHAP_PANELS]:
        if not panels[key]:
            continue
        ax = _next()
        shap_results = _get_shap_results(key)
        if shap_results is None:
            _missing(ax, f'{key} shap_vals')
        else:
            heatmap_panel(ax, shap_results.temporal_feature,
                          sample.feature_names, lf,
                          f"{key.replace('_', ' ').capitalize()} SHAP")

    # PLOT TEMPORAL SHAP
    unit = ' (relative)' if normalise else ''

    def _temporal_series(key: str) -> np.ndarray | None:
        result = _get_shap_results(key)
        if result is None or result.temporal is None:
            return None
        values = result.temporal
        if result.segment_boundaries is not None and len(values) != lf:
            values = spread_segments(values, result.segment_boundaries)
        return values

    if panels['temporal']:
        ax = _next()
        series = {k: _temporal_series(k) for k in SHAP_PANELS}
        temporal_panel(ax, series, 'Temporal', n_frames=lf)

    if panels['feature']:
        ax = _next()
        series = {
            k: (sample.feature_names, _get_shap_results(k).feature)
            for k in SHAP_PANELS
        }
        feature_panel(ax, series, 'Feature')
        # series = {
        #     k: (sample.feature_names, _norm(sample.ground_truth_feature(k)))
        #     for k in sample.ground_truth
        # }
        # if sample.model_map is not None:
        #     series['Model'] = (sample.model_feature_names,
        #                        _norm(sample.model_feature_from_map()))
        # if series:
        #     feature_panel(ax, series,
        #                   f'Feature importance{unit} (separate game)')
        # else:
        #     _missing(ax, 'feature importance')

    if panels['outputs']:
        ax = _next()
        pass
        # if root is None:
        #     _missing(ax, 'output metadata')
        # else:
        #     outputs_panel(
        #         ax,
        #         root['output_type'],
        #         ground_truth=horizon_outputs(sample.horizon_hazard,
        #                                      root['output_type'],
        #                                      root['hazard_bins']),
        #         model_prediction=sample.model_prediction,
        #         hazard_bins=root['hazard_bins'],
        #         horizon=root['horizon'],
        #         death_offset=(None if sample.death_frame is None else
        #                       sample.death_frame - lf),
        #     )


def plot_sample(
    h5_path: Path | str,
    sample_idx: int | None,
    title: str | None = None,
    panels: dict | None = None,
) -> plt.Figure:
    """One row of panels for a single stored sample."""
    panels = _resolve_panels(panels)
    n_panels = len(panels)
    fig, axes = plt.subplots(1,
                             n_panels,
                             figsize=(5.2 * n_panels, 4),
                             squeeze=False)
    plot_sample_on_axes(h5_path, list(axes[0]), sample_idx, panels=panels)
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def sample_indices(h5_path: Path | str) -> list[int]:
    """Numerically sorted sample ids."""
    with h5py.File(h5_path, 'r') as f:
        if 'Signals' in f:
            return []  # single-sample file written at the root
        return sorted(int(k) for k in f.keys())


def plot_samples(
    h5_path: Path | str,
    save_dir: Path | str,
    num_plot_samples: int,
    title_prefix: str = '',
    panels: dict | None = None,
) -> list[Path]:
    """Save one figure per stored sample into ``save_dir``."""
    h5_path = Path(h5_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = sample_indices(h5_path)
    indices = indices[:num_plot_samples] if indices else [None]

    saved = []
    for sample_idx in indices:
        label = 'sample' if sample_idx is None else f'sample {sample_idx}'
        title = f'{title_prefix} — {label}' if title_prefix else label
        fig = plot_sample(h5_path, sample_idx, title=title, panels=panels)
        name = ('sample.png'
                if sample_idx is None else f'sample_{sample_idx:02d}.png')
        path = save_dir / name
        fig.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        saved.append(path)
    log.info(f'Saved {len(saved)} sample figures to {save_dir}')
    return saved


def plot_dataset_average(
    h5_path: Path | str,
    title: str | None = None,
    normalise: bool = True,
) -> plt.Figure:
    """Mean |SHAP| over every stored sample: heatmaps, then temporal and feature bars.

    Signed values are turned to absolute value *before* averaging: across many
    samples a feature/frame that's sometimes protective and sometimes harmful
    would otherwise cancel towards zero and look unimportant, which is the
    opposite of what a dataset-level summary should show (per-sample panels in
    ``plot_sample_on_axes`` keep the sign instead, since there it's what makes
    a single explanation readable).

    Samples have different landmark frames, so maps/curves are stacked on a
    shared absolute frame axis and NaN-padded before averaging; the shaded
    band on the temporal panel is how many samples reach each frame — the
    right-hand columns average over just the handful of long-``lf`` samples
    and are correspondingly noisy.

    ``normalise`` (default) rescales each series to sum to 1 after averaging,
    so estimators are compared on *relative* importance — the baseline choice
    (mask vs. distributional) and the model's larger output range otherwise
    leave one estimator visually dominant.
    """
    h5_path = Path(h5_path)
    indices = sample_indices(h5_path)
    indices = indices if indices else [None]
    samples = [SampleWithSHAP.read_h5(h5_path, i) for i in indices]

    keys = [k for k in PANEL_ORDER if k in SHAP_PANELS]
    max_lf = max(s.landmark_frame for s in samples)
    feature_names = samples[0].feature_names

    def _get(sample: SampleWithSHAP, key: str) -> SHAPResult | None:
        explainer = [e for e in sample.explainers() if e in key]
        background = [b for b in sample.backgrounds() if b in key]
        if len(explainer) == 0 or len(background) == 0:
            return None
        log.debug(f'SHAP val keys: {sample.shap_vals.keys()}')
        return sample.shap_vals[(explainer[0], background[0])]

    def _norm(values: np.ndarray) -> np.ndarray:
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    def _missing(ax: plt.Axes, what: str) -> None:
        ax.text(0.5,
                0.5,
                f'No {what} in\n{h5_path.name}',
                ha='center',
                va='center',
                fontsize=8,
                transform=ax.transAxes)
        ax.set_axis_off()

    unit = 'relative' if normalise else 'mean |SHAP|'
    fig, axes = plt.subplots(1,
                             len(keys) + 2,
                             figsize=(6 * (len(keys) + 2), 4),
                             squeeze=False)
    axes = list(axes[0])

    # PLOT TEMPORAL_FEATURE HEATMAPS
    for key in keys:
        ax = axes.pop(0)
        stack = np.full((len(samples), len(feature_names), max_lf), np.nan)
        for i, sample in enumerate(samples):
            result = _get(sample, key)
            if result is None or result.temporal_feature is None:
                continue
            lf = sample.landmark_frame
            tf = result.temporal_feature
            if result.segment_boundaries is not None and tf.shape[-1] != lf:
                tf = spread_segments(tf, result.segment_boundaries)
            # Kernel SHAP's rows only cover the model's own input features
            # (e.g. no Hazard), a subset/reorder of sample.feature_names —
            # align by name rather than assuming row counts match.
            row_lookup = dict(zip(sample.feature_names, tf))
            for r, name in enumerate(feature_names):
                row = row_lookup.get(name)
                if row is not None:
                    stack[i, r, :lf] = np.abs(row[:lf])
        if np.all(np.isnan(stack)):
            _missing(ax, f'{key} shap_vals')
            continue
        heatmap_panel(ax,
                      _norm(np.nanmean(stack, axis=0)),
                      feature_names,
                      max_lf,
                      f"{key.replace('_', ' ').capitalize()}  {unit}",
                      diverging=False)

    # PLOT TEMPORAL SHAP
    temporal_series = {}
    support = np.zeros(max_lf)
    for j, key in enumerate(keys):
        stack = np.full((len(samples), max_lf), np.nan)
        for i, sample in enumerate(samples):
            result = _get(sample, key)
            if result is None or result.temporal is None:
                continue
            values = result.temporal
            if (result.segment_boundaries is not None
                    and len(values) != sample.landmark_frame):
                values = spread_segments(values, result.segment_boundaries)
            lf = sample.landmark_frame
            stack[i, :lf] = np.abs(values[:lf])
            if j == 0:
                support[:lf] += 1
        temporal_series[key] = (None if np.all(np.isnan(stack)) else _norm(
            np.nanmean(stack, axis=0)))
    ax = axes.pop(0)
    if all(v is None for v in temporal_series.values()):
        _missing(ax, 'temporal importance')
    else:
        temporal_panel(ax,
                       temporal_series,
                       f'Temporal importance  {unit}',
                       n_frames=max_lf,
                       support=support)

    # PLOT FEATURE SHAP
    feature_series = {}
    for key in keys:
        rows = []
        for sample in samples:
            result = _get(sample, key)
            if result is None or result.feature is None:
                continue
            lookup = dict(zip(sample.feature_names, np.abs(result.feature)))
            rows.append([lookup.get(n, np.nan) for n in feature_names])
        values = (np.full(len(feature_names), np.nan)
                  if not rows else np.nanmean(np.asarray(rows), axis=0))
        feature_series[key] = (feature_names, _norm(values))
    ax = axes.pop(0)
    if all(np.all(np.isnan(v)) for _, v in feature_series.values()):
        _missing(ax, 'feature importance')
    else:
        feature_panel(ax, feature_series, f'Feature importance  {unit}')

    suptitle = f'Dataset average over {len(samples)} samples'
    if title:
        suptitle += f' — {title}'
    fig.suptitle(suptitle, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
