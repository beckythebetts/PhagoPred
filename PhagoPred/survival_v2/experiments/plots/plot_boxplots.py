from __future__ import annotations
from dataclasses import fields

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()

_LOWER_IS_BETTER = {'MSE', 'Brier_Score'}

_BOXPLOT_PROPS = dict(
    showmeans=True,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=1.5),
    meanprops=dict(marker='o',
                   markerfacecolor='black',
                   markeredgecolor='black',
                   markersize=4),
    whiskerprops=dict(color='black', linewidth=1.5),
    capprops=dict(color='black', linewidth=1.5),
)


def plot_box_plots(
        all_experiments: list[ExperimentRecord],
        varying_params: dict) -> plt.Figure | tuple[plt.Figure, plt.Figure]:
    """Plot box plots of results metrics"""
    if len(varying_params) == 1:
        return _plot_box_plots_1var(all_experiments, varying_params)
    if len(varying_params) == 2:
        return _plot_box_plots_2var(all_experiments, varying_params)
    else:
        log.info(
            f'Skipping plotting box plots, {len(varying_params)} varying parameters.'
        )
        return None


def _plot_box_plots_1var(all_experiments: list[ExperimentRecord],
                         varying_param: dict) -> plt.Figure:
    """Plot box plots for one varying parameter."""
    _sample_results = all_experiments[0].results
    metrics = [
        m.name for m in fields(_sample_results)
        if isinstance(getattr(_sample_results, m.name), float)
    ]
    varying_param_name = list(varying_param.keys())[0]
    varying_param_values = varying_param[varying_param_name]
    fig, axs = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))

    labels = [str(var.name) for var in varying_param_values]
    x = np.arange(len(labels))
    cmap = plt.get_cmap('Set1')

    for i, metric in enumerate(metrics):
        for j, (var) in enumerate(varying_param_values):
            experiments = [
                experiment for experiment in all_experiments if getattr(
                    experiment.experiemnt_cfg, varying_param_name) == var
            ]
            data = [
                getattr(experiment.results, metric)
                for experiment in experiments
            ]
            axs[i].boxplot(
                data,
                positions=[j],
                boxprops=dict(facecolor=cmap(j),
                              edgecolor='black',
                              linewidth=1.5),
                **_BOXPLOT_PROPS,
            )
            axs[i].set_xlabel(varying_param_name.capitalize().replace(
                '_', ' '),
                              fontsize=12)
            axs[i].set_ylabel(metric.replace('_', ' '), fontsize=12)
            axs[i].set_xticks(x, labels, ha='center')
            axs[i].set_title(f"{metric.replace('_', ' ')}",
                             fontsize=14,
                             fontweight='bold')
        if metric in _LOWER_IS_BETTER:
            axs[i].invert_yaxis()

    plt.tight_layout()
    return fig


def _plot_box_plots_2var(
        all_experiments: list[ExperimentRecord],
        varying_params: dict) -> tuple[plt.Figure, plt.Figure]:
    """Return two figures for 2 varying parameters, each with the axes swapped."""
    param_name_1, param_name_2 = tuple(varying_params.keys())
    fig1 = _plot_box_plots_2var_single(all_experiments,
                                       varying_params,
                                       x_param=param_name_1,
                                       color_param=param_name_2)
    fig2 = _plot_box_plots_2var_single(all_experiments,
                                       varying_params,
                                       x_param=param_name_2,
                                       color_param=param_name_1)
    return fig1, fig2


def _plot_box_plots_2var_single(all_experiments: list[ExperimentRecord],
                                varying_params: dict, x_param: str,
                                color_param: str) -> plt.Figure:
    """Plot box plots for 2 varying parameters.

    x_param values form the x-axis groups.
    color_param values are shown as side-by-side coloured boxes within each group.
    """
    _sample_results = all_experiments[0].results
    metrics = [
        m.name for m in fields(_sample_results)
        if isinstance(getattr(_sample_results, m.name), float)
    ]

    x_vals = varying_params[x_param]
    color_vals = varying_params[color_param]
    n_x = len(x_vals)
    n_color = len(color_vals)

    width = 0.8 / n_color
    x = np.arange(n_x)
    cmap = plt.get_cmap('Set1')
    log.info(f'Num metrics = {len(metrics)}')
    fig, axs = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))
    if len(metrics) == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        for k, cval in enumerate(color_vals):
            positions = []
            data = []
            for j, xval in enumerate(x_vals):
                experiments = [
                    exp for exp in all_experiments
                    if getattr(exp.experiemnt_cfg, x_param) == xval
                    and getattr(exp.experiemnt_cfg, color_param) == cval
                ]
                metric_data = [
                    getattr(exp.results, metric) for exp in experiments
                ]
                if metric_data:
                    offset = (k - n_color / 2 + 0.5) * width
                    positions.append(x[j] + offset)
                    data.append(metric_data)

            if data:
                axs[i].boxplot(
                    data,
                    positions=positions,
                    widths=width * 0.8,
                    boxprops=dict(facecolor=cmap(k),
                                  edgecolor='black',
                                  linewidth=1.5),
                    **_BOXPLOT_PROPS,
                )

        x_labels = [v.name for v in x_vals]
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels, rotation=0, ha='center')
        axs[i].set_xlabel(x_param.replace('_', ' ').capitalize(), fontsize=12)
        axs[i].set_ylabel(metric.replace('_', ' '), fontsize=12)
        axs[i].set_title(metric.replace('_', ' '),
                         fontsize=14,
                         fontweight='bold')
        axs[i].grid(True, axis='y')
        if metric in _LOWER_IS_BETTER:
            axs[i].invert_yaxis()

    handles = [
        Patch(facecolor=cmap(k),
              edgecolor='black',
              label=val.name,
              linewidth=1.5) for k, val in enumerate(color_vals)
    ]
    axs[-1].legend(
        handles=handles,
        title=color_param.replace('_', ' ').capitalize(),
        fontsize=10,
        loc='best',
        frameon=True,
    )

    plt.tight_layout()
    return fig
