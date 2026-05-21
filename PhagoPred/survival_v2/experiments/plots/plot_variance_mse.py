from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.survival_v2.evaluate import BinaryResults, SurvivalResults
from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord
from .utils import plot_med_range_on_ax

log = get_logger()


def plot_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | None:
    """Plot model MSE against ground-truth variance bounds (floor/ceiling).

    For survival experiments: per-bin hazard MSE vs hazard variance.
    For binary experiments: CDF MSE at prediction horizon vs CDF variance.
    Returns None if no variance data is available.
    """
    if not all_experiments:
        return None

    is_binary = isinstance(all_experiments[0].results, BinaryResults)

    if is_binary:
        return _plot_binary_variance_mse(all_experiments, varying_params)
    else:
        return _plot_survival_variance_mse(all_experiments, varying_params)


# ── survival ──────────────────────────────────────────────────────────────────

def _plot_survival_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | None:
    # Find reference variance from the first experiment that has it
    ref_var = next(
        (e.variances for e in all_experiments
         if e.variances is not None and 'hazard_total' in e.variances),
        None,
    )
    if ref_var is None:
        log.info('No hazard variance data found — skipping variance/MSE plot')
        return None

    hazard_total = np.asarray(ref_var['hazard_total'])
    hazard_unobserved = np.asarray(ref_var['hazard_unobserved'])
    hazard_bins = np.asarray(ref_var['hazard_bins'])
    # x-axis: bin midpoints
    bin_mids = 0.5 * (hazard_bins[:-1] + hazard_bins[1:])

    # Filter to experiments that have per-bin MSE
    valid = [e for e in all_experiments if getattr(e.results, 'hazard_mse_per_bin', None) is not None]
    if not valid:
        log.info('No hazard_mse_per_bin in results — skipping variance/MSE plot')
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('Set1')

    # Variance bands (same across all groups — draw once)
    ax.fill_between(bin_mids, hazard_unobserved, hazard_total,
                    color='grey', alpha=0.25, label='Unexplained variance (total − unobserved)')
    ax.plot(bin_mids, hazard_total, color='grey', linewidth=1.5,
            linestyle='--', label='Total variance (ceiling)')
    ax.plot(bin_mids, hazard_unobserved, color='grey', linewidth=1.5,
            linestyle=':', label='Unobserved variance (floor)')

    if varying_params:
        var_par_name = list(varying_params.keys())[0]
        var_par_vals = varying_params[var_par_name]
        for i, val in enumerate(var_par_vals):
            group = [e for e in valid
                     if getattr(e.experiemnt_cfg, var_par_name) == val]
            if not group:
                continue
            mses = [np.asarray(e.results.hazard_mse_per_bin) for e in group]
            label = getattr(val, 'name', str(val))
            plot_med_range_on_ax(ax, bin_mids, mses, cmap(i), (0, 100), label)
    else:
        mses = [np.asarray(e.results.hazard_mse_per_bin) for e in valid]
        plot_med_range_on_ax(ax, bin_mids, mses, cmap(0), (0, 100), 'Model MSE')

    ax.set_xlabel('Bin midpoint (frames)', fontsize=12)
    ax.set_ylabel('Hazard MSE', fontsize=12)
    ax.set_title('Per-bin hazard MSE vs variance bounds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', frameon=True)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


# ── binary ─────────────────────────────────────────────────────────────────────

def _plot_binary_variance_mse(
    all_experiments: list[ExperimentRecord],
    varying_params: dict,
) -> plt.Figure | None:
    ref_var = next(
        (e.variances for e in all_experiments
         if e.variances is not None and 'cdf_total' in e.variances),
        None,
    )
    if ref_var is None:
        log.info('No CDF variance data found — skipping variance/MSE plot')
        return None

    cdf_total = np.asarray(ref_var['cdf_total'])
    cdf_unobserved = np.asarray(ref_var['cdf_unobserved'])

    # Get prediction horizon from dataset config (first experiment)
    prediction_horizon = getattr(
        all_experiments[0].experiemnt_cfg.dataset, 'prediction_horizon', None)
    if prediction_horizon is None:
        log.info('No prediction_horizon in dataset config — skipping variance/MSE plot')
        return None

    horizon_idx = int(prediction_horizon)
    if horizon_idx >= len(cdf_total):
        horizon_idx = len(cdf_total) - 1

    ceil_val = float(cdf_total[horizon_idx])
    floor_val = float(cdf_unobserved[horizon_idx])

    valid = [e for e in all_experiments
             if getattr(e.results, 'true_cdf_mse', None) is not None]
    if not valid:
        log.info('No true_cdf_mse in results — skipping variance/MSE plot')
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('Set1')

    ax.axhline(ceil_val, color='grey', linewidth=1.5, linestyle='--',
               label=f'Total variance (ceiling) = {ceil_val:.4f}')
    ax.axhline(floor_val, color='grey', linewidth=1.5, linestyle=':',
               label=f'Unobserved variance (floor) = {floor_val:.4f}')
    ax.axhspan(floor_val, ceil_val, color='grey', alpha=0.15)

    if varying_params:
        var_par_name = list(varying_params.keys())[0]
        var_par_vals = varying_params[var_par_name]
        x_positions = np.arange(len(var_par_vals))
        mse_vals = []
        labels = []
        colors = []
        for i, val in enumerate(var_par_vals):
            group = [e for e in valid
                     if getattr(e.experiemnt_cfg, var_par_name) == val]
            if not group:
                continue
            mses = [e.results.true_cdf_mse for e in group]
            mse_med = float(np.median(mses))
            mse_lo = float(np.percentile(mses, 0))
            mse_hi = float(np.percentile(mses, 100))
            label = getattr(val, 'name', str(val))
            mse_vals.append((mse_med, mse_lo, mse_hi))
            labels.append(label)
            colors.append(cmap(i))

        for i, (med, lo, hi) in enumerate(mse_vals):
            ax.scatter([i], [med], color=colors[i], zorder=5, s=60,
                       label=f'{labels[i]}: {med:.4f}')
            ax.plot([i, i], [lo, hi], color=colors[i], linewidth=2)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
        ax.set_xlabel(var_par_name.replace('_', ' ').capitalize(), fontsize=12)
    else:
        mses = [e.results.true_cdf_mse for e in valid]
        med = float(np.median(mses))
        lo = float(np.percentile(mses, 0))
        hi = float(np.percentile(mses, 100))
        ax.scatter([0], [med], color=cmap(0), zorder=5, s=60,
                   label=f'Model MSE: {med:.4f}')
        ax.plot([0, 0], [lo, hi], color=cmap(0), linewidth=2)
        ax.set_xticks([0])
        ax.set_xticklabels(['Model'])

    ax.set_ylabel('CDF MSE', fontsize=12)
    ax.set_title(
        f'CDF MSE vs variance bounds (horizon={prediction_horizon})',
        fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', frameon=True)
    ax.grid(True, alpha=0.4, axis='y')
    fig.tight_layout()
    return fig
