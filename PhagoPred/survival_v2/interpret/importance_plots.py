"""Reusable panel primitives for SHAP importance figures.

Kept free of torch so the cross-model plots under ``experiments.plots`` can use
the same panels as the per-experiment plots in ``ground_truth_importance``.

Convention throughout: per-sample panels show *signed* SHAP values, because the
sign is what makes a single explanation readable. Dataset averages show
``mean |SHAP|``, because signed values from different samples cancel and the
average would understate importance.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.interpret.importance_data import (
    dataset_average,
    horizon_outputs,
    load_sample_importances,
    read_root_attrs_optional,
    sample_indices,
)

log = get_logger()

# Series whose feature axis is the model's, not the graph's (no Hazard node).
MODEL_KEYS = ('Model', )

SERIES_STYLE = {
    'Interventional': dict(color='tab:blue', lw=1.2),
    'Observational': dict(color='tab:green', lw=1.2),
    'Model': dict(color='tab:red', lw=1.2),
}


def series_rows(key: str, feature_names: list[str],
                model_feature_names: list[str] | None) -> list[str]:
    """Feature axis a given series lives on."""
    if key in MODEL_KEYS and model_feature_names is not None:
        return model_feature_names
    return feature_names


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

    The x extent is always the frame window, so panels line up column-wise even
    when computed on different grids (per-frame ground truth vs per-segment model
    SHAP). ``row_normalise`` scales each row to its own max, which the signals
    panel needs because Hazard and the features live on different scales.
    ``diverging=False`` switches to a sequential map for non-negative averages.
    """
    m = np.asarray(matrix, dtype=float)
    if row_normalise:
        scale = np.nanmax(np.abs(m), axis=1, keepdims=True)
        m = m / np.where(scale == 0.0, 1.0, scale)
    extent = [0, n_frames, -0.5, m.shape[0] - 0.5]
    if diverging:
        vmax = max(float(np.nanmax(np.abs(m))), 1e-9)
        im = ax.imshow(m,
                       aspect='auto',
                       origin='lower',
                       cmap='RdBu_r',
                       vmin=-vmax,
                       vmax=vmax,
                       extent=extent)
    else:
        im = ax.imshow(m,
                       aspect='auto',
                       origin='lower',
                       cmap='Greys',
                       vmin=0.0,
                       vmax=max(float(np.nanmax(m)), 1e-9),
                       extent=extent)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('frame', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)


def temporal_panel(
    ax: plt.Axes,
    series: dict[str, np.ndarray],
    title: str,
    n_frames: int | None = None,
    support: np.ndarray | None = None,
) -> None:
    """Per-frame importance curves, one line per estimator.

    Lines rather than bars: with landmark frames up to ~400, several overlaid bar
    series are unreadable. ``support`` (dataset averages only) shades how many
    samples reach each frame — the right-hand columns average over the handful of
    long-``lf`` samples and are correspondingly noisy.
    """
    for name, values in series.items():
        if values is None:
            continue
        ax.plot(np.arange(len(values)),
                values,
                label=name,
                **SERIES_STYLE.get(name, {}))
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
) -> None:
    """Grouped per-feature bars; series may cover different feature sets.

    Ground truth carries a Hazard bar the model has no counterpart for. It is
    left as a gap rather than dropped, so the feature axis stays aligned across
    series and across models.
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
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(positions + offset,
               heights,
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


# ─────────────────────────── figures ───────────────────────────
# The panel primitives above are assembled into whole figures here, straight from
# an h5 file of stored SHAP samples. Both interpret entry points share this:
#
#   * ground_truth_importance -> shap_samples.h5: Interventional / Observational
#     ground-truth maps, root scenario/horizon/hazard-bin metadata, model SHAP,
#     model prediction. All seven panels.
#   * run_interpret -> SHAP.h5: model SHAP only, no causal graph to compare to.
#     Pass ``MODEL_ONLY_PANELS`` to draw just signals + model + temporal + feature.
#
# Panels whose data is absent from the file are annotated, never raised, so the
# same call works on either file.

PANEL_ORDER = ('signals', 'interventional', 'observational', 'model',
               'temporal', 'feature', 'outputs')
DEFAULT_PANELS = {name: True for name in PANEL_ORDER}
MODEL_ONLY_PANELS = {
    'signals': True,
    'interventional': False,
    'observational': False,
    'model': True,
    'temporal': True,
    'feature': True,
    'outputs': False,
}


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

    Panels are consumed in the order signals, interventional, observational,
    model, temporal, feature, outputs — one axis per enabled panel (see
    ``PANEL_ORDER`` / ``DEFAULT_PANELS``). The heatmaps share a [0, lf) frame
    axis so they can be read column-wise against each other; model SHAP is
    per-segment and is spread onto that frame grid.

    The temporal and feature panels collapse the model's joint (segment x
    feature) map. Ground-truth temporal curves are summed over the model's
    features only, so both sides sum over the same set; Hazard still appears in
    the heatmaps and the per-feature bars.

    ``sample_idx=None`` reads a single-sample file written at the h5 root. Panels
    whose data is absent are annotated rather than raising.
    """
    flags = _resolve_panels(panels)
    enabled = [flags[name] for name in PANEL_ORDER]
    assert sum(enabled) == len(axes), (
        f'{sum(enabled)} panels enabled but {len(axes)} axes given')

    h5_path = Path(h5_path)
    root = read_root_attrs_optional(h5_path)
    sample = load_sample_importances(h5_path, sample_idx)

    lf = sample.landmark_frame
    death_str = ('censored' if sample.death_frame is None else
                 f'{sample.death_frame:.0f}')
    remaining = list(axes)

    def _next() -> plt.Axes:
        return remaining.pop(0)

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

    if flags['signals']:
        ax = _next()
        if sample.signals is None:
            _missing(ax, 'signals')
        else:
            heatmap_panel(
                ax, sample.signals, sample.feature_names, lf,
                f'Signals (row-normalised)  lf={lf}  death={death_str}',
                row_normalise=True)

    for key in ('Interventional', 'Observational'):
        if not flags[key.lower()]:
            continue
        ax = _next()
        if key not in sample.ground_truth:
            _missing(ax, f'{key} importances')
        else:
            heatmap_panel(ax, sample.ground_truth[key], sample.feature_names,
                          lf, f'{key} GT SHAP')

    if flags['model']:
        ax = _next()
        if sample.model_map is None:
            _missing(ax, 'Model SHAP')
        else:
            n_segments = len(sample.segment_boundaries) - 1
            heatmap_panel(
                ax, sample.model_map, sample.model_feature_names, lf,
                f'Model KernelSHAP ({n_segments} segments, per-frame)')

    unit = ' (relative)' if normalise else ''
    if flags['temporal']:
        ax = _next()
        series = {
            k: _norm(sample.ground_truth_temporal(k))
            for k in sample.ground_truth
        }
        if sample.model_map is not None:
            series['Model'] = _norm(sample.model_temporal_from_map())
        if series:
            # Feature names can be long; don't spell them into the title (it
            # overflows and collides with the neighbouring panel titles).
            temporal_panel(
                ax, series,
                f'Temporal importance{unit} '
                f'(summed over {len(sample.shared_feature_names)} features)',
                n_frames=lf)
        else:
            _missing(ax, 'temporal importance')

    if flags['feature']:
        ax = _next()
        series = {
            k: (sample.feature_names, _norm(sample.ground_truth_feature(k)))
            for k in sample.ground_truth
        }
        if sample.model_map is not None:
            series['Model'] = (sample.model_feature_names,
                               _norm(sample.model_feature_from_map()))
        if series:
            feature_panel(ax, series,
                          f'Feature importance{unit} (separate game)')
        else:
            _missing(ax, 'feature importance')

    if flags['outputs']:
        ax = _next()
        if root is None:
            _missing(ax, 'output metadata')
        else:
            outputs_panel(
                ax,
                root['output_type'],
                ground_truth=horizon_outputs(sample.horizon_hazard,
                                             root['output_type'],
                                             root['hazard_bins']),
                model_prediction=sample.model_prediction,
                hazard_bins=root['hazard_bins'],
                horizon=root['horizon'],
                death_offset=(None if sample.death_frame is None else
                              sample.death_frame - lf),
            )


def plot_sample(
    h5_path: Path | str,
    sample_idx: int | None,
    title: str | None = None,
    panels: dict | None = None,
) -> plt.Figure:
    """One row of panels for a single stored sample."""
    flags = _resolve_panels(panels)
    n_panels = sum(flags[name] for name in PANEL_ORDER)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4),
                             squeeze=False)
    plot_sample_on_axes(h5_path, list(axes[0]), sample_idx, panels=flags)
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


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

    Samples have different landmark frames, so maps are stacked on an absolute
    frame axis and NaN-padded; the shaded band on the temporal panel is how many
    samples reach each frame. The right-hand columns average over the handful of
    long-lf samples and are correspondingly noisy.

    ``normalise`` (default) rescales each estimator's values in each panel to sum
    to 1, so model and ground truth are compared on *relative* importance — the
    baseline choice (mask vs. distributional) and the model's larger output range
    otherwise leave the model 2-3x the ground-truth scale and visually dominant.

    One heatmap per stored estimator: ``{Interventional, Observational, Model}``
    for a ground-truth file, just ``Model`` for run_interpret's SHAP.h5.
    """
    h5_path = Path(h5_path)
    average = dataset_average(h5_path)

    def norm(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    unit = 'relative' if normalise else 'mean |SHAP|'
    map_keys = list(average.maps)
    fig, axes = plt.subplots(1,
                             len(map_keys) + 2,
                             figsize=(6 * (len(map_keys) + 2), 4),
                             squeeze=False)
    axes = list(axes[0])

    for key in map_keys:
        rows = series_rows(key, average.feature_names,
                           average.model_feature_names)
        heatmap_panel(axes.pop(0),
                      norm(average.maps[key]),
                      rows,
                      average.max_landmark_frame,
                      f'{key}  {unit}',
                      diverging=False)

    temporal_panel(axes.pop(0),
                   {k: norm(v) for k, v in average.temporal.items()},
                   f'Temporal importance  {unit}',
                   support=average.support)

    feature_series = {}
    for key, values in average.feature.items():
        rows = series_rows(key, average.feature_names,
                           average.model_feature_names)
        feature_series[key] = (rows, norm(values))
    feature_panel(axes.pop(0), feature_series, f'Feature importance  {unit}')

    suptitle = f'Dataset average over {average.num_samples} samples'
    if title:
        suptitle += f' — {title}'
    fig.suptitle(suptitle, fontweight='bold', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
