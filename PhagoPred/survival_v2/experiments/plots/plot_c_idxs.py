from __future__ import annotations
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord
from .utils import plot_med_range_on_ax

log = get_logger()


def plot_c_idxs_scores(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
    percentile_range: Tuple[int, int] = (0, 100)
) -> plt.Figure | Tuple[plt.Figure, plt.Figure]:
    """Plot ROC curves"""
    if len(varying_params) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _plot_c_idx_on_ax(axs, all_experiments, varying_params,
                          percentile_range)
        return fig
    if len(varying_params) == 2:
        param_name_1, param_name_2 = tuple(varying_params.keys())
        param_vals_1 = varying_params[param_name_1]
        param_vals_2 = varying_params[param_name_2]
        fig1 = _plot_c_idx_2var(
            all_experiments,
            param_name_1,
            param_vals_1,
            param_name_2,
            param_vals_2,
            percentile_range,
        )
        fig2 = _plot_c_idx_2var(
            all_experiments,
            param_name_2,
            param_vals_2,
            param_name_1,
            param_vals_1,
            percentile_range,
        )
        return fig1, fig2
    else:
        log.info(
            f'Skipping plotting C-Index curves, {len(varying_params)} varying paramaters'
        )
        return None


def _plot_c_idx_2var(
    all_experiments: list[ExperimentRecord],
    outer_param_name: str,
    outer_vals: list,
    inner_param_name: str,
    inner_vals: list,
    percentile_range: Tuple[int, int],
) -> plt.Figure:

    fig, axs = plt.subplots(
        1,
        len(outer_vals),
        figsize=(6 * len(outer_vals), 6),
    )
    for i, outer_val in enumerate(outer_vals):
        experiments = [
            experiment for experiment in all_experiments if getattr(
                experiment.experiemnt_cfg, outer_param_name) == outer_val
        ]
        _plot_c_idx_on_ax(axs[i], experiments, {inner_param_name: inner_vals},
                          percentile_range)

        axs[i].set_title(f'{outer_param_name}={outer_val.name}', fontsize=14)
        fig.suptitle('Concordance Index', fontsize=14, fontweight='bold')
        if not i == len(outer_vals) - 1:
            axs[i].legend().set_visible(False)

    return fig


def _plot_c_idx_on_ax(
    ax: plt.Axes,
    all_experiments: list[ExperimentRecord],
    varying_param: dict,
    percentile_range: Tuple[int, int] = (0, 100)
) -> plt.Figure:
    """Plot median +- specified range of reciever operatir characteristic curves fo experiments."""
    var_par_name = list(varying_param.keys())[0]
    var_par_vals = varying_param[var_par_name]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.get_cmap('Set1')
    # max_time = max([
    #     max(experiment.results.brier_times) for experiment in all_experiments
    # ])
    time_vals = np.arange(all_experiments[0].experiemnt_cfg.dataset.num_bins)
    for i, val in enumerate(var_par_vals):
        experiments = [
            experiment for experiment in all_experiments
            if getattr(experiment.experiemnt_cfg, var_par_name) == val
        ]

        # interpolated_brier = [
        #     np.interp(time_vals, experiment.results.brier_times,
        #               experiment.results.brier_scores)
        #     for experiment in experiments
        # ]
        c_idxs = [
            experiment.results.c_index_per_bin for experiment in experiments
        ]

        plot_med_range_on_ax(ax, time_vals, c_idxs, cmap(i), percentile_range,
                             val.name)

    ax.set_title('Concordance Index', fontsize=14, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Concordance Index', fontsize=12)
    ax.grid(True)
    ax.legend(title=var_par_name.replace('_', ' ').capitalize(),
              fontsize=10,
              loc='best',
              frameon=True)
