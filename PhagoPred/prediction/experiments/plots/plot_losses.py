from __future__ import annotations
from typing import Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord
from .utils import plot_med_range_on_ax

log = get_logger()


def plot_experiment_losses(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
    percentile_range: Tuple[int, int] = (0, 100)
) -> plt.Figure | Tuple[plt.Figure, plt.Figure]:
    """Plot experiment losses for one or two varying parameters"""

    if len(varying_params) == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        _plot_experiment_losses_on_ax(axs, all_experiments, varying_params,
                                      percentile_range)
        return fig
    if len(varying_params) == 2:
        param_name_1, param_name_2 = tuple(varying_params.keys())
        param_vals_1 = varying_params[param_name_1]
        param_vals_2 = varying_params[param_name_2]
        fig1 = _plot_losses_2var(
            all_experiments,
            param_name_1,
            param_vals_1,
            param_name_2,
            param_vals_2,
            percentile_range,
        )
        fig2 = _plot_losses_2var(
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
            f'Skipping plotting losses, {len(varying_params)} varying paramaters'
        )
        return None


def _plot_losses_2var(
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
        _plot_experiment_losses_on_ax(axs[i], experiments,
                                      {inner_param_name: inner_vals},
                                      percentile_range)
        axs[i].set_title(
            f"{outer_param_name.replace('_', ' ').capitalize()}={outer_val.name}",
            fontsize=14)
        fig.suptitle('Validation Losses', fontsize=14, fontweight='bold')
        if not i == len(outer_vals) - 1:
            axs[i].legend().set_visible(False)
    return fig


def _plot_experiment_losses_on_ax(
    ax: plt.Axes,
    all_experiments: list[ExperimentRecord],
    varying_param: dict,
    percentile_range: Tuple[int, int] = (0, 100)) -> None:
    """Plot losses for all experiments"""
    var_par_name = list(varying_param.keys())[0]
    var_par_vals = varying_param[var_par_name]
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    cmap = plt.get_cmap('Set1')
    for i, val in enumerate(var_par_vals):
        experiments = [
            experiment for experiment in all_experiments
            if getattr(experiment.experiemnt_cfg, var_par_name) == val
        ]

        val_losses = [
            _get_experiment_losses(experiment, 'val')
            for experiment in experiments
        ]
        val_losses = [
            val_loss for val_loss in val_losses if val_loss is not None
        ]

        if len(val_losses) == 0:
            continue

        plot_med_range_on_ax(
            ax,
            np.arange(len(val_losses[0])),
            val_losses,
            cmap(i),
            percentile_range,
            val.name,
        )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss Curves', fontsize=14)
    ax.legend(title=var_par_name.replace('_', ' ').capitalize(),
              fontsize=10,
              loc='best',
              frameon=True)
    ax.grid(True, alpha=0.3)


def _get_experiment_losses(experiemnt: ExperimentRecord,
                           loss_type: Literal['train', 'val']) -> list[float]:
    """Get a np.array of losses for a given epxeriment.
    Return
    ------
        Losses: list
    """
    losses = []
    if not isinstance(experiemnt.training_history, list):
        return None
    for row in experiemnt.training_history:
        losses.append(row[loss_type]['total'])
    return losses
