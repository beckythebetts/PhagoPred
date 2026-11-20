"""How closely model SHAP matches ground-truth SHAP, per scenario, model,
background and attribution level.

Per sample, for each level (feature, temporal, temporal x feature) and each
background explained for both predictors:

    corr     Pearson correlation of model vs GT attributions (shape)
    rel_err  ||model - GT|| / ||GT||                          (shape + scale)
    scale    <model, GT> / <GT, GT>: model magnitude relative to GT (1 = same)
    gt_noise ||GT stderr|| / ||GT||: the GT's own Monte Carlo error, a floor
             for rel_err

Experiments on one scenario are compared on the samples they all have.
Written to the suite directory:

    shap_agreement_samples.csv   one row per (experiment, background, level,
                                 sample)
    shap_agreement.csv           median and IQR over samples
    shap_agreement.png           medians with IQR bars
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.configs.io import load_experiment_cfg
from PhagoPred.survival_v2.interpret.data_models import (BackgroundEnum,
                                                         PredictorEnum)
from PhagoPred.survival_v2.interpret.plots import (_BG_COLOUR, read_samples,
                                                   sample_indices)

log = get_logger()

SHAP_FILE = 'shap.h5'
LEVELS = ('feature', 'temporal', 'temporal_feature')
LEVEL_LABEL = {
    'feature': 'Feature',
    'temporal': 'Temporal',
    'temporal_feature': 'Temporal × feature',
}
METRICS = ('corr', 'rel_err', 'scale', 'gt_noise')
_MODEL_MARKER = ('o', 's', '^', 'D')


def _metrics(model: np.ndarray, gt: np.ndarray,
             gt_err: np.ndarray | None) -> dict[str, float]:
    m, g = np.ravel(model), np.ravel(gt)
    gg = float(g @ g)
    if gg == 0.0:
        return dict.fromkeys(METRICS, np.nan)
    corr = (np.corrcoef(m, g)[0, 1]
            if m.std() > 0 and g.std() > 0 else np.nan)
    return dict(corr=float(corr),
                rel_err=float(np.linalg.norm(m - g) / np.sqrt(gg)),
                scale=float(m @ g / gg),
                gt_noise=(float(np.linalg.norm(gt_err) / np.sqrt(gg))
                          if gt_err is not None else np.nan))


def _experiments(suite_dir: Path) -> list[dict]:
    out = []
    for d in sorted(suite_dir.iterdir()):
        if not (d / SHAP_FILE).exists() or not (d / 'config.json').exists():
            continue
        cfg = load_experiment_cfg(d / 'config.json')
        if not cfg.dataset.shap_paths:
            continue
        out.append(
            dict(experiment=d.name,
                 path=d / SHAP_FILE,
                 scenario=Path(cfg.dataset.shap_paths[0]).parent.name,
                 model=cfg.model.model_type))
    return out


def agreement_samples(suite_dir: Path | str) -> pd.DataFrame:
    """One row per (experiment, background, level, sample)."""
    by_scenario = defaultdict(list)
    for e in _experiments(Path(suite_dir)):
        by_scenario[e['scenario']].append(e)

    rows = []
    for scenario, exps in by_scenario.items():
        common = set(sample_indices(exps[0]['path']))
        for e in exps[1:]:
            common &= set(sample_indices(e['path']))
        common = sorted(common)
        for e in exps:
            for idx, s in zip(common, read_samples(e['path'], common)):
                for bg in BackgroundEnum:
                    gt = s.shap_vals.get((PredictorEnum.GROUND_TRUTH, bg))
                    model = s.shap_vals.get((PredictorEnum.MODEL, bg))
                    if gt is None or model is None:
                        continue
                    for level in LEVELS:
                        g, m = getattr(gt, level), getattr(model, level)
                        if g is None or m is None:
                            continue
                        rows.append(
                            dict(scenario=scenario,
                                 model=e['model'],
                                 experiment=e['experiment'],
                                 background=bg.value,
                                 level=level,
                                 sample=idx,
                                 **_metrics(m, g,
                                            getattr(gt, level + '_stderr'))))
    return pd.DataFrame(rows)


def summarise(samples: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR over samples of each metric."""
    keys = ['scenario', 'model', 'experiment', 'background', 'level']
    g = samples.groupby(keys, sort=False)
    out = g.size().rename('n_samples').to_frame()
    for metric in METRICS:
        col = g[metric]
        out[f'{metric}_median'] = col.median()
        out[f'{metric}_q25'] = col.quantile(0.25)
        out[f'{metric}_q75'] = col.quantile(0.75)
    return out.reset_index()


def plot_agreement(summary: pd.DataFrame) -> plt.Figure:
    """Rows: correlation, relative error. Columns: attribution level.
    x: scenario; colour: background; marker: model; bars: IQR over samples.
    Grey dashes on the error row: the GT's own noise (the floor)."""
    scenarios = list(dict.fromkeys(summary['scenario']))
    models = list(dict.fromkeys(summary['model']))
    bgs = [b for b in BackgroundEnum if b.value in set(summary['background'])]
    combos = [(m, b) for m in models for b in bgs]
    width = 0.8 / max(len(combos), 1)

    fig, axes = plt.subplots(2,
                             len(LEVELS),
                             figsize=(5 * len(LEVELS), 7),
                             sharex=True,
                             squeeze=False)
    for col, level in enumerate(LEVELS):
        lv = summary[summary['level'] == level]
        for row, metric in enumerate(('corr', 'rel_err')):
            ax = axes[row, col]
            for k, (model, bg) in enumerate(combos):
                d = lv[(lv['model'] == model) & (lv['background'] == bg.value)]
                d = d.set_index('scenario').reindex(scenarios)
                x = np.arange(len(scenarios)) + (k - (len(combos) - 1) / 2) * width
                med = d[f'{metric}_median'].to_numpy(float)
                err = np.abs(
                    np.stack([d[f'{metric}_q25'], d[f'{metric}_q75']]) - med)
                ax.errorbar(x,
                            med,
                            yerr=err,
                            fmt=_MODEL_MARKER[models.index(model) %
                                              len(_MODEL_MARKER)],
                            color=_BG_COLOUR[bg],
                            ms=8,
                            lw=1.5,
                            capsize=0,
                            mec='white',
                            mew=1.0)
            if metric == 'rel_err':
                noise = lv.groupby('scenario')['gt_noise_median'].median()
                for i, sc in enumerate(scenarios):
                    if sc in noise and np.isfinite(noise[sc]):
                        ax.hlines(noise[sc], i - 0.45, i + 0.45, color='0.6',
                                  ls='--', lw=1.2)
                ax.set_ylim(bottom=0)
            else:
                ax.axhline(1.0, color='0.85', lw=0.8, zorder=0)
            ax.grid(axis='y', color='0.92', lw=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(LEVEL_LABEL[level], fontweight='bold')
            if col == 0:
                ax.set_ylabel('Correlation with GT' if metric == 'corr' else
                              'Relative error ‖model − GT‖ / ‖GT‖')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels([s.replace('_', '\n') for s in scenarios],
                               fontsize=8)

    handles = [
        Line2D([], [], ls='', marker=_MODEL_MARKER[i % len(_MODEL_MARKER)],
               color='0.3', ms=8, label=m) for i, m in enumerate(models)
    ] + [
        Line2D([], [], ls='', marker='o', color=_BG_COLOUR[b], ms=8,
               label=b.value.replace('_', ' ')) for b in bgs
    ] + [Line2D([], [], color='0.6', ls='--', label='GT noise (floor)')]
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle('Model vs ground-truth SHAP (median, IQR over samples)',
                 fontweight='bold', y=0.995)
    return fig


def write_agreement(suite_dir: Path | str) -> pd.DataFrame | None:
    """Write the CSVs and figure (module docstring) to ``suite_dir``."""
    suite_dir = Path(suite_dir)
    samples = agreement_samples(suite_dir)
    if samples.empty:
        log.warning(f'{suite_dir}: no GT + model SHAP to compare')
        return None
    summary = summarise(samples)
    samples.to_csv(suite_dir / 'shap_agreement_samples.csv', index=False)
    summary.to_csv(suite_dir / 'shap_agreement.csv', index=False)
    fig = plot_agreement(summary)
    fig.savefig(suite_dir / 'shap_agreement.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    log.info(f'SHAP agreement written to {suite_dir}')
    return summary
