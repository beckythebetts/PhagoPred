from typing import Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.configs import ExperimentCfg
from .experiment_record_dataclass import ExperimentRecord

log = get_logger()


def plot_confusion_matrices(
        all_experiments: list[ExperimentRecord],
        varying_params: dict,
        cm_type: Literal['cm_expected', 'cm_argmax'],
        percentile_range: Tuple[int, int] = (0, 100),
):
    """PLot confusion matrices for one or two varying params"""
    if len(varying_params) == 1:
        return _plt_cms_1var(all_experiments, varying_params, cm_type,
                             percentile_range)
    if len(varying_params) == 2:
        return _plot_cms_2var(all_experiments, varying_params, cm_type,
                              percentile_range)
    else:
        log.info(
            f'Skipping plotting confusion matrices, {len(varying_params)} varying paramaters'
        )
        return None


def _normalise_cm(cm: np.ndarray) -> np.ndarray:
    row_sums = np.sum(cm, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def _plot_cm_on_ax(
    ax: plt.Axes,
    experiments: list[ExperimentRecord],
    cm_type: Literal['cm_expected', 'cm_argmax'],
    percentile_range: Tuple[int, int] = (0, 1)
) -> None:
    if not hasattr(experiments[0].results, cm_type):
        cm_type = 'cm'
    cms = np.array([
        _normalise_cm(getattr(experiment.results, cm_type))
        for experiment in experiments
    ])
    cm_median = np.median(cms, axis=0)
    cm_low = np.percentile(cms, percentile_range[0], axis=0)
    cm_high = np.percentile(cms, percentile_range[1], axis=0)

    num_bins = cm_median.shape[0]
    ax.imshow(cm_median, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('True', fontsize=10, fontweight='bold')
    ax.set_xticks(np.arange(num_bins))
    ax.set_yticks(np.arange(num_bins))

    if num_bins == 2:
        bin_names = ['No Event', 'Event']
    else:
        bin_names = [f'Bin {i}' for i in range(num_bins)]
    ax.set_xticklabels(bin_names)
    ax.set_yticklabels(bin_names, rotation=90, va='center')
    ax.tick_params(axis='both', length=0)

    for k in range(num_bins):
        for l in range(num_bins):
            text_colour = 'white' if cm_median[k, l] > 0.5 else 'black'
            ax.text(l,
                    k,
                    f'{cm_median[k, l]:.2f}',
                    ha='center',
                    va='center',
                    color=text_colour,
                    fontsize=9,
                    fontweight='bold')
            ax.text(l,
                    k + 0.25,
                    f'({cm_low[k, l]:.2f}, {cm_high[k, l]:.2f})',
                    ha='center',
                    va='center',
                    color=text_colour,
                    fontsize=7)


def _plt_cms_1var(
    all_experiments: list[ExperimentRecord],
    varying_param: dict,
    cm_type: Literal['cm_expected', 'cm_argmax'],
    percentile_range: Tuple[int, int] = (0, 100)
) -> plt.Figure:
    """Plot confusion matrices for one varying parameter, table-style."""
    var_par_name = list(varying_param.keys())[0]
    var_par_vals = varying_param[var_par_name]
    num_bins = all_experiments[0].experiemnt_cfg.dataset.num_bins
    num_bins = 2 if num_bins == 1 else num_bins
    n_vals = len(var_par_vals)
    fig, axs = plt.subplots(1,
                            n_vals,
                            figsize=(num_bins * n_vals, num_bins + 0.5))
    if n_vals == 1:
        axs = [axs]

    for i, val in enumerate(var_par_vals):
        experiments = [
            experiment for experiment in all_experiments
            if getattr(experiment.experiemnt_cfg, var_par_name) == val
        ]
        _plot_cm_on_ax(axs[i], experiments, cm_type, percentile_range)
        # Column header: just the value name
        axs[i].set_title(val.name, fontsize=12, fontweight='bold')
        # Only show y-axis label and ticks on leftmost plot
        if i > 0:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])

    param_label = var_par_name.replace('_', ' ').capitalize()
    fig.suptitle(param_label, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def _plot_cms_2var(
    all_experiments: list[ExperimentCfg],
    varying_params: dict,
    cm_type: Literal['cm_expected', 'cm_argmax'],
    percentile_range: Tuple = (0, 100)
) -> plt.Figure:
    """Plot confusion matrices for two varying params, table-style."""
    param_name_1, param_name_2 = tuple(varying_params.keys())
    param_vals_1 = varying_params[param_name_1]
    param_vals_2 = varying_params[param_name_2]
    n_var1 = len(param_vals_1)
    n_var2 = len(param_vals_2)

    num_bins = all_experiments[0].experiemnt_cfg.dataset.num_bins
    num_bins = 2 if num_bins == 1 else num_bins
    fig, axs = plt.subplots(n_var1,
                            n_var2,
                            figsize=(num_bins * n_var2 + 0.5,
                                     num_bins * n_var1 + 0.5))

    for i, val1 in enumerate(param_vals_1):
        for j, val2 in enumerate(param_vals_2):
            experiments = [
                exp for exp in all_experiments
                if getattr(exp.experiemnt_cfg, param_name_1) == val1
                and getattr(exp.experiemnt_cfg, param_name_2) == val2
            ]
            _plot_cm_on_ax(axs[i, j], experiments, cm_type, percentile_range)
            # Column header: top row only
            if i == 0:
                axs[i, j].set_title(val2.name, fontsize=12, fontweight='bold')
            # Row header: left column only, prepended to ylabel
            if j == 0:
                axs[i, j].set_ylabel(f'{val1.name}\n\nTrue',
                                     fontsize=10,
                                     fontweight='bold')
            else:
                axs[i, j].set_ylabel('')
                axs[i, j].set_yticklabels([])
            # x-axis label and ticks: bottom row only
            if i < n_var1 - 1:
                axs[i, j].set_xlabel('')
                axs[i, j].set_xticklabels([])

    # Shared axis labels for param names
    col_label = param_name_2.replace('_', ' ').capitalize()
    row_label = param_name_1.replace('_', ' ').capitalize()
    fig.suptitle(col_label, fontsize=13, fontweight='bold')
    fig.supylabel(row_label, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.00, wspace=0.05)
    return fig
