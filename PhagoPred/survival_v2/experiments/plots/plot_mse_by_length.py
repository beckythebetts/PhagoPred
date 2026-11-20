from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.survival_v2.evaluate import BinaryResults
from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()

# Fixed categorical order (colour-vision-deficiency validated); never cycled.
_COLOURS = [
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
    '#4a3aa7', '#e34948'
]


def plot_cdf_mse_by_length(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | None:
    """cdf_mse vs sequence length at the landmark for binary experiments.

    One panel per dataset (if the dataset varies), one colour per combination
    of the other varying parameters, one line per experiment (repeats drawn
    separately). Total and unobserved variance are shown as horizontal lines.
    """
    valid = [
        e for e in all_experiments if isinstance(e.results, BinaryResults)
        and e.results.cdf_mse_by_length is not None
    ]
    if not valid:
        log.info('No cdf_mse_by_length in results — skipping plot')
        return None

    panel_param = 'dataset' if 'dataset' in varying_params else None
    colour_params = [p for p in varying_params if p != panel_param]
    panel_vals = varying_params[panel_param] if panel_param else [None]

    def colour_key(e: ExperimentRecord) -> str:
        return ', '.join(
            getattr(getattr(e.experiemnt_cfg, p), 'name',
                    str(getattr(e.experiemnt_cfg, p)))
            for p in colour_params) or 'Model'

    colour_keys = list(dict.fromkeys(colour_key(e) for e in valid))
    if len(colour_keys) > len(_COLOURS):
        log.info(f'Skipping cdf_mse_by_length plot: {len(colour_keys)} '
                 f'series exceeds {len(_COLOURS)} colours')
        return None
    colours = dict(zip(colour_keys, _COLOURS))

    fig, axes = plt.subplots(1,
                             len(panel_vals),
                             figsize=(4.8 * len(panel_vals) + 1, 4.5),
                             squeeze=False)  # own y per panel: scales differ
    for ax, pval in zip(axes[0], panel_vals):
        exps = [
            e for e in valid if panel_param is None
            or getattr(e.experiemnt_cfg, panel_param) == pval
        ]
        for e in exps:
            by_len = e.results.cdf_mse_by_length
            width = by_len['bin_width']
            pts = [(s + width / 2, m)
                   for s, m in zip(by_len['bin_starts'], by_len['cdf_mse'])
                   if m is not None]
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs,
                        ys,
                        color=colours[colour_key(e)],
                        linewidth=2,
                        label=colour_key(e))

        totals = [
            e.results.total_true_variance for e in exps
            if e.results.total_true_variance is not None
        ]
        unobs = [
            e.results.unobserved_true_variance for e in exps
            if e.results.unobserved_true_variance is not None
        ]
        if totals:
            ax.axhline(float(np.median(totals)),
                       color='dimgray',
                       linestyle='--',
                       linewidth=1.5,
                       label='Total variance')
        if unobs:
            ax.axhline(float(np.median(unobs)),
                       color='black',
                       linestyle=':',
                       linewidth=2,
                       label='Unobserved variance')

        if pval is not None:
            ax.set_title(getattr(pval, 'name', str(pval)),
                         loc='left',
                         fontsize=12)
        ax.set_xlabel('Sequence length at landmark (frames)', fontsize=11)
        # Bottom at 0; top fits every curve and the total-variance line.
        tops = [
            float(np.max(l.get_ydata())) for l in ax.get_lines()
            if len(l.get_ydata())
        ]
        ax.set_ylim(0, 1.08 * max(tops) if tops else None)
        ax.grid(True, axis='y', alpha=0.3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel('cdf_mse', fontsize=11)

    handles, labels = [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles,
               labels,
               loc='upper center',
               ncol=min(len(labels), 4),
               frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.suptitle('Prediction error vs sequence length',
                 x=0.01,
                 ha='left',
                 y=1.15,
                 fontsize=14,
                 fontweight='bold')
    fig.tight_layout()
    return fig
