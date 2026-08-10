"""Reusable panel primitives for SHAP importance figures.

Kept free of torch so the cross-model plots under ``experiments.plots`` can use
the same panels as the per-experiment plots in ``ground_truth_importance``.

Convention throughout: per-sample panels show *signed* SHAP values, because the
sign is what makes a single explanation readable. Dataset averages show
``mean |SHAP|``, because signed values from different samples cancel and the
average would understate importance.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
    ax.set_xticklabels(names, fontsize=8)
    ax.axhline(0.0, color='black', lw=0.5, alpha=0.4)
    ax.set_ylabel('importance', fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)
