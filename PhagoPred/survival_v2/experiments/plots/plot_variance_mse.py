from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

from PhagoPred.survival_v2.evaluate import BinaryResults, SurvivalResults
from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord
from .utils import plot_med_range_on_ax

log = get_logger()

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


def plot_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | None:
    """Plot model MSE against ground-truth variance bounds (floor/ceiling).

    For survival experiments: per-bin hazard MSE vs hazard variance.
    For binary experiments: CDF MSE vs CDF variance bounds, one graph.
    Returns None if no variance data is available.
    """
    if not all_experiments:
        return None

    is_binary = isinstance(all_experiments[0].results, BinaryResults)

    if is_binary:
        return _plot_binary_variance_mse(all_experiments, varying_params)
    else:
        return _plot_survival_variance_mse(all_experiments, varying_params)


# ===BINARY===


def _plot_binary_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | tuple[plt.Figure, plt.Figure] | None:
    valid = [e for e in all_experiments if e.results.cdf_mse is not None]
    if not valid:
        log.info('No cdf_mse in binary results — skipping variance/MSE plot')
        return None

    if len(varying_params) == 1:
        param_name = list(varying_params.keys())[0]
        return _plot_binary_variance_mse_single(valid,
                                                varying_params,
                                                x_param=param_name,
                                                color_param=None)
    elif len(varying_params) == 2:
        p1, p2 = tuple(varying_params.keys())
        fig1 = _plot_binary_variance_mse_single(valid,
                                                varying_params,
                                                x_param=p1,
                                                color_param=p2)
        fig2 = _plot_binary_variance_mse_single(valid,
                                                varying_params,
                                                x_param=p2,
                                                color_param=p1)
        return fig1, fig2
    else:
        log.info(
            f'Skipping binary variance/MSE plot, {len(varying_params)} varying parameters.'
        )
        return None


def _plot_binary_variance_mse_single(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
    x_param: str,
    color_param: str | None,
) -> plt.Figure:
    x_vals = varying_params[x_param]
    color_vals = varying_params[color_param] if color_param else [None]
    n_x = len(x_vals)
    n_color = len(color_vals)

    width = 0.8 / n_color
    x = np.arange(n_x)
    cmap = plt.get_cmap('Set2')

    fig, ax = plt.subplots(figsize=(max(6, 2 * n_x * n_color), 6))

    for k, cval in enumerate(color_vals):
        positions, cdf_mse_data = [], []
        total_var_pts, unobs_var_pts = [], []

        for j, xval in enumerate(x_vals):
            if cval is not None:
                exps = [
                    e for e in all_experiments
                    if getattr(e.experiemnt_cfg, x_param) == xval
                    and getattr(e.experiemnt_cfg, color_param) == cval
                ]
            else:
                exps = [
                    e for e in all_experiments
                    if getattr(e.experiemnt_cfg, x_param) == xval
                ]

            mses = [
                e.results.cdf_mse for e in exps
                if e.results.cdf_mse is not None
            ]
            if not mses:
                continue

            offset = (k - n_color / 2 + 0.5) * width
            pos = float(x[j]) + offset
            positions.append(pos)
            cdf_mse_data.append(mses)

            tot = [
                e.results.total_true_variance for e in exps
                if e.results.total_true_variance is not None
            ]
            unobs = [
                e.results.unobserved_true_variance for e in exps
                if e.results.unobserved_true_variance is not None
            ]
            total_var_pts.append((pos, float(np.median(tot))) if tot else None)
            unobs_var_pts.append((pos,
                                  float(np.median(unobs))) if unobs else None)

        if not cdf_mse_data:
            continue

        colour = cmap(k)
        ax.boxplot(
            cdf_mse_data,
            positions=positions,
            widths=width * 0.8,
            boxprops=dict(facecolor=colour, edgecolor='black', linewidth=1.5),
            **_BOXPLOT_PROPS,
        )

        tot_valid = [(p, v) for pt in total_var_pts if pt for p, v in [pt]]
        unobs_valid = [(p, v) for pt in unobs_var_pts if pt for p, v in [pt]]

        label_prefix = (getattr(cval, 'name', str(cval)) +
                        ' — ') if cval is not None else ''
        if tot_valid:
            ps, vs = zip(*tot_valid)
            ax.scatter(ps,
                       vs,
                       marker='x',
                       s=300,
                       color='red',
                       linewidths=2.5,
                       zorder=5,
                       label=f'{label_prefix}Total variance')
        if unobs_valid:
            ps, vs = zip(*unobs_valid)
            ax.scatter(ps,
                       vs,
                       marker='_',
                       s=80,
                       color='blue',
                       linewidths=2,
                       zorder=5,
                       label=f'{label_prefix}Unobserved variance')

    x_labels = [getattr(v, 'name', str(v)) for v in x_vals]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.set_xlabel(x_param.replace('_', ' ').capitalize(), fontsize=12)
    ax.set_ylabel('CDF MSE', fontsize=12)
    ax.set_title('CDF MSE vs Variance Bounds', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.4)

    legend_handles = []
    if color_param:
        legend_handles += [
            Patch(facecolor=cmap(k),
                  edgecolor='black',
                  label=getattr(v, 'name', str(v)),
                  linewidth=1.5) for k, v in enumerate(color_vals)
        ]
    legend_handles += [
        Line2D([0], [0],
               marker='_',
               color='grey',
               linestyle='None',
               markersize=14,
               markeredgewidth=2.5,
               label='Total variance'),
        Line2D([0], [0],
               marker='x',
               color='grey',
               linestyle='None',
               markersize=8,
               markeredgewidth=2,
               label='Unobserved variance'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='best', frameon=True)
    plt.tight_layout()
    return fig


# ===SURVIVAL===

# Metric -> (total variance, unobserved variance, per-bin MSE) result attributes
_SURVIVAL_METRICS = {
    'hazard': ('total_true_hazard_variance', 'unobserved_true_hazard_variance',
               'hazard_mse_per_bin', 'Hazard'),
    'pmf': ('total_true_pmf_variance', 'unobserved_true_pmf_variance',
            'pmf_mse_per_bin', 'PMF'),
}


def _plot_survival_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | tuple[plt.Figure, ...] | None:
    if len(varying_params) == 1:
        plot_fn = _plt_survival_1_var
    elif len(varying_params) == 2:
        plot_fn = _plt_survival_2_var
    else:
        log.info(
            f'Skipping plotting box plots, {len(varying_params)} varying parameters.'
        )
        return None

    figs = [
        fig for metric in _SURVIVAL_METRICS
        if (fig := plot_fn(all_experiments, varying_params, metric)
            ) is not None
    ]
    if not figs:
        return None
    return tuple(figs) if len(figs) > 1 else figs[0]


def _gather_metric(experiments: list[ExperimentRecord],
                   metric: str) -> tuple[list, list, list] | None:
    """Collect (total, unobserved, mse) per-bin arrays for a metric.

    Returns None if any of the three is unavailable for these experiments.
    """
    total_attr, unobs_attr, mse_attr, _ = _SURVIVAL_METRICS[metric]
    total = [getattr(e.results, total_attr) for e in experiments]
    unobserved = [getattr(e.results, unobs_attr) for e in experiments]
    mse = [getattr(e.results, mse_attr) for e in experiments]
    if any(v is None for v in total + unobserved + mse):
        return None
    return total, unobserved, mse


def _plt_survival_1_var(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
    metric: str = 'hazard',
) -> plt.Figure | None:
    var_name = list(varying_params.keys())[0]
    var_items = varying_params[var_name]
    fig, axs = plt.subplots(1,
                            len(var_items),
                            figsize=(6 * len(var_items), 6),
                            squeeze=False)
    axs = axs[0]
    has_data = False
    for i, var_val in enumerate(var_items):
        experiments = [
            experiment for experiment in all_experiments
            if getattr(experiment.experiemnt_cfg, var_name) == var_val
        ]
        gathered = _gather_metric(experiments, metric)
        axs[i].set_title(f'{var_name} = {var_val.name}',
                         fontsize=14,
                         fontweight='bold')
        if gathered is None:
            continue
        has_data = True
        _plot_vars_on_ax(axs[i], *gathered)

    if not has_data:
        plt.close(fig)
        return None
    axs[0].legend()
    fig.suptitle(f'{_SURVIVAL_METRICS[metric][3]} MSE vs Variance Bounds',
                 fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    return fig


def _plt_survival_2_var(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
    metric: str = 'hazard',
) -> plt.Figure | None:
    keys = list(varying_params.keys())
    row_name, col_name = keys[0], keys[1]
    row_vals = varying_params[row_name]
    col_vals = varying_params[col_name]

    n_rows, n_cols = len(row_vals), len(col_vals)
    fig, axs = plt.subplots(n_rows,
                            n_cols,
                            figsize=(6 * n_cols, 6 * n_rows),
                            squeeze=False)

    has_data = False
    for r, row_val in enumerate(row_vals):
        for c, col_val in enumerate(col_vals):
            experiments = [
                experiment for experiment in all_experiments
                if getattr(experiment.experiemnt_cfg, row_name) == row_val
                and getattr(experiment.experiemnt_cfg, col_name) == col_val
            ]
            ax = axs[r, c]
            ax.set_title(
                f'{row_name}={row_val.name}, {col_name}={col_val.name}',
                fontsize=12,
                fontweight='bold',
            )
            gathered = _gather_metric(experiments, metric)
            if gathered is None:
                continue
            has_data = True
            _plot_vars_on_ax(ax, *gathered)

    if not has_data:
        plt.close(fig)
        return None
    axs[0, 0].legend()
    fig.suptitle(f'{_SURVIVAL_METRICS[metric][3]} MSE vs Variance Bounds',
                 fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_vars_on_ax(ax: plt.Axes, true: list[np.ndarray],
                     unobserved: list[np.ndarray],
                     mse: list[np.ndarray]) -> None:

    def _get_mean_std(x: list[np.ndarray]):
        return np.mean(x, axis=0), np.std(x, axis=0)

    true_mean, true_std = _get_mean_std(true)
    unobserved_mean, unobserved_std = _get_mean_std(unobserved)
    mean_mse, unobserved_mse = _get_mean_std(mse)

    num_bins = len(true_mean)

    # log.info('True mean shape: ')

    def _plot(means: np.ndarray, stds: np.ndarray, colour: tuple | str,
              linestyle: str, label: str):
        ax.plot(
            np.arange(num_bins),
            means,
            color=colour,
            marker=' ',
            # fmt=" ",
            linestyle=linestyle,
            label=label)
        ax.fill_between(
            np.arange(num_bins),
            means - stds,
            means + stds,
            color=colour,
            alpha=0.3,
            edgecolor='none',
        )

    _plot(true_mean, true_std, 'red', linestyle='--', label='Total Varaince')
    _plot(unobserved_mean,
          unobserved_std,
          colour='k',
          linestyle='--',
          label='Variance due to unobserved time steps')
    _plot(mean_mse,
          unobserved_mse,
          'blue',
          linestyle='-',
          label='Mean Squared Error')


# # ===SURVIVAl====

# def _plot_survival_variance_mse(
#     all_experiments: list[ExperimentRecord],
#     varying_params: dict,
# ) -> plt.Figure | None:
#     # Find reference variance from the first experiment that has it
#     ref_var = next(
#         (e.variances for e in all_experiments
#          if e.variances is not None and 'hazard_total' in e.variances),
#         None,
#     )
#     if ref_var is None:
#         log.info('No hazard variance data found — skipping variance/MSE plot')
#         return None

#     hazard_total = np.asarray(ref_var['hazard_total'])
#     hazard_unobserved = np.asarray(ref_var['hazard_unobserved'])
#     hazard_bins = np.asarray(ref_var['hazard_bins'])
#     # x-axis: bin midpoints
#     bin_mids = 0.5 * (hazard_bins[:-1] + hazard_bins[1:])

#     # Aggregate frame-level variance to bin-level if shapes don't match
#     if len(hazard_total) != len(bin_mids):
#         hazard_total = np.array([
#             hazard_total[int(hazard_bins[i]):int(hazard_bins[i + 1])].mean()
#             for i in range(len(bin_mids))
#         ])
#         hazard_unobserved = np.array([
#             hazard_unobserved[int(hazard_bins[i]):int(hazard_bins[i +
#                                                                   1])].mean()
#             for i in range(len(bin_mids))
#         ])

#     # Filter to experiments that have per-bin MSE
#     valid = [
#         e for e in all_experiments
#         if getattr(e.results, 'hazard_mse_per_bin', None) is not None
#     ]
#     if not valid:
#         log.info(
#             'No hazard_mse_per_bin in results — skipping variance/MSE plot')
#         return None

#     fig, ax = plt.subplots(figsize=(8, 5))
#     cmap = plt.get_cmap('Set1')

#     # Variance bands (same across all groups — draw once)
#     ax.fill_between(bin_mids,
#                     hazard_unobserved,
#                     hazard_total,
#                     color='grey',
#                     alpha=0.25,
#                     label='Unexplained variance (total − unobserved)')
#     ax.plot(bin_mids,
#             hazard_total,
#             color='grey',
#             linewidth=1.5,
#             linestyle='--',
#             label='Total variance (ceiling)')
#     ax.plot(bin_mids,
#             hazard_unobserved,
#             color='grey',
#             linewidth=1.5,
#             linestyle=':',
#             label='Unobserved variance (floor)')

#     if varying_params:
#         var_par_name = list(varying_params.keys())[0]
#         var_par_vals = varying_params[var_par_name]
#         for i, val in enumerate(var_par_vals):
#             group = [
#                 e for e in valid
#                 if getattr(e.experiemnt_cfg, var_par_name) == val
#             ]
#             if not group:
#                 continue
#             mses = [np.asarray(e.results.hazard_mse_per_bin) for e in group]
#             label = getattr(val, 'name', str(val))
#             plot_med_range_on_ax(ax, bin_mids, mses, cmap(i), (0, 100), label)
#     else:
#         mses = [np.asarray(e.results.hazard_mse_per_bin) for e in valid]
#         plot_med_range_on_ax(ax, bin_mids, mses, cmap(0), (0, 100),
#                              'Model MSE')

#     ax.set_xlabel('Bin midpoint (frames)', fontsize=12)
#     ax.set_ylabel('Hazard MSE', fontsize=12)
#     ax.set_title('Per-bin hazard MSE vs variance bounds',
#                  fontsize=14,
#                  fontweight='bold')
#     ax.legend(fontsize=9, loc='best', frameon=True)
#     ax.grid(True, alpha=0.4)
#     fig.tight_layout()
#     return fig

# # ===BINARY===

# def _plot_binary_variance_mse(
#     all_experiments: list[ExperimentRecord],
#     varying_params: dict,
# ) -> plt.Figure | None:
#     ref_var = next(
#         (e.variances for e in all_experiments
#          if e.variances is not None and 'cdf_total' in e.variances),
#         None,
#     )
#     if ref_var is None:
#         log.info('No CDF variance data found — skipping variance/MSE plot')
#         return None

#     cdf_total = np.asarray(ref_var['cdf_total'])
#     cdf_unobserved = np.asarray(ref_var['cdf_unobserved'])

#     # Get prediction horizon from dataset config (first experiment)
#     prediction_horizon = getattr(all_experiments[0].experiemnt_cfg.dataset,
#                                  'prediction_horizon', None)
#     if prediction_horizon is None:
#         log.info(
#             'No prediction_horizon in dataset config — skipping variance/MSE plot'
#         )
#         return None

#     horizon_idx = int(prediction_horizon)
#     if horizon_idx >= len(cdf_total):
#         horizon_idx = len(cdf_total) - 1

#     ceil_val = float(cdf_total[horizon_idx])
#     floor_val = float(cdf_unobserved[horizon_idx])

#     valid = [
#         e for e in all_experiments
#         if getattr(e.results, 'true_cdf_mse', None) is not None
#     ]
#     if not valid:
#         log.info('No true_cdf_mse in results — skipping variance/MSE plot')
#         return None

#     fig, ax = plt.subplots(figsize=(7, 5))
#     cmap = plt.get_cmap('Set1')

#     ax.axhline(ceil_val,
#                color='grey',
#                linewidth=1.5,
#                linestyle='--',
#                label=f'Total variance (ceiling) = {ceil_val:.4f}')
#     ax.axhline(floor_val,
#                color='grey',
#                linewidth=1.5,
#                linestyle=':',
#                label=f'Unobserved variance (floor) = {floor_val:.4f}')
#     ax.axhspan(floor_val, ceil_val, color='grey', alpha=0.15)

#     if varying_params:
#         var_par_name = list(varying_params.keys())[0]
#         var_par_vals = varying_params[var_par_name]
#         x_positions = np.arange(len(var_par_vals))
#         mse_vals = []
#         labels = []
#         colors = []
#         for i, val in enumerate(var_par_vals):
#             group = [
#                 e for e in valid
#                 if getattr(e.experiemnt_cfg, var_par_name) == val
#             ]
#             if not group:
#                 continue
#             mses = [e.results.true_cdf_mse for e in group]
#             mse_med = float(np.median(mses))
#             mse_lo = float(np.percentile(mses, 0))
#             mse_hi = float(np.percentile(mses, 100))
#             label = getattr(val, 'name', str(val))
#             mse_vals.append((mse_med, mse_lo, mse_hi))
#             labels.append(label)
#             colors.append(cmap(i))

#         for i, (med, lo, hi) in enumerate(mse_vals):
#             ax.scatter([i], [med],
#                        color=colors[i],
#                        zorder=5,
#                        s=60,
#                        label=f'{labels[i]}: {med:.4f}')
#             ax.plot([i, i], [lo, hi], color=colors[i], linewidth=2)

#         ax.set_xticks(range(len(labels)))
#         ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
#         ax.set_xlabel(var_par_name.replace('_', ' ').capitalize(), fontsize=12)
#     else:
#         mses = [e.results.true_cdf_mse for e in valid]
#         med = float(np.median(mses))
#         lo = float(np.percentile(mses, 0))
#         hi = float(np.percentile(mses, 100))
#         ax.scatter([0], [med],
#                    color=cmap(0),
#                    zorder=5,
#                    s=60,
#                    label=f'Model MSE: {med:.4f}')
#         ax.plot([0, 0], [lo, hi], color=cmap(0), linewidth=2)
#         ax.set_xticks([0])
#         ax.set_xticklabels(['Model'])

#     ax.set_ylabel('CDF MSE', fontsize=12)
#     ax.set_title(f'CDF MSE vs variance bounds (horizon={prediction_horizon})',
#                  fontsize=14,
#                  fontweight='bold')
#     ax.legend(fontsize=9, loc='best', frameon=True)
#     ax.grid(True, alpha=0.4, axis='y')
#     fig.tight_layout()
#     return fig
