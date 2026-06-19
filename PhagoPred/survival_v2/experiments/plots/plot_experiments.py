from __future__ import annotations
from pathlib import Path
import json
from typing import Literal, Callable
from dataclasses import asdict, fields
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.survival_v2.configs.io import ExperimentCfg, load_experiment_cfg
from PhagoPred.survival_v2.evaluate import BinaryResults, SurvivalResults
from PhagoPred.utils.logger import get_logger
from .experiment_record_dataclass import ExperimentRecord
from .plot_boxplots import plot_box_plots
from .plot_losses import plot_experiment_losses
from .cm_plots import plot_confusion_matrices
from .roc_plots import plot_rocs
from .plot_brier_scores import plot_brier_scores
from .plot_c_idxs import plot_c_idxs_scores
from .plot_variance_mse import plot_variance_mse

log = get_logger()

plt.rcParams['font.family'] = 'serif'


def plot_experiment_results(experiments_dir: Path,
                            ignore_params: list[str] | None = None) -> None:
    """Plot metrics for all experiments in a directory."""
    all_experiemnts = []
    varying_params = {f.name: [] for f in fields(ExperimentCfg)}
    for experiment_path in experiments_dir.iterdir():
        if experiment_path.is_dir():
            config = experiment_path / 'config.json'
            results = experiment_path / 'evaluation_results.json'
            training_history = experiment_path / 'training_history.json'

            assert config.exists() and results.exists(
            ), f"Experiemnt path {experiment_path} doesn't contain config/results .jsons"

            config = load_experiment_cfg(config)

            is_binary = config.dataset.num_bins == 1

            with results.open('r') as f:
                results = json.load(f)
                if is_binary:
                    results = BinaryResults(**results)
                else:
                    results = SurvivalResults(**results)

            with training_history.open('r') as f:
                training_history = json.load(f)

            # variances = None
            # variances_path = experiment_path / 'variances.npz'
            # if variances_path.exists():
            #     npz = np.load(variances_path)
            #     variances = {k: npz[k] for k in npz.files}

            all_experiemnts.append(
                ExperimentRecord(config, results, training_history))
            log.info(f'Gathered resulsts for experiment {asdict(config)}')
            for f in fields(ExperimentCfg):
                val = getattr(config, f.name)
                if val not in varying_params[f.name]:
                    varying_params[f.name].append(val)

    varying_params = {k: v for k, v in varying_params.items() if len(v) > 1}
    if ignore_params:
        varying_params = {
            k: v
            for k, v in varying_params.items() if k not in ignore_params
        }
    log.info(f'Got {len(varying_params)} varying paramaters {varying_params}')

    _plot_and_save(plot_box_plots, all_experiemnts, varying_params,
                   experiments_dir / 'results.png')
    _plot_and_save(plot_experiment_losses, all_experiemnts, varying_params,
                   experiments_dir / 'losses.png')
    _plot_and_save(partial(plot_confusion_matrices, cm_type='cm_expected'),
                   all_experiemnts, varying_params,
                   experiments_dir / 'confusion_matrices_expected.png')
    _plot_and_save(partial(plot_confusion_matrices, cm_type='cm_argmax'),
                   all_experiemnts, varying_params,
                   experiments_dir / 'confusion_matrices_argmax.png')
    if is_binary:
        _plot_and_save(plot_rocs, all_experiemnts, varying_params,
                       experiments_dir / 'ROC.png')
    else:
        _plot_and_save(plot_brier_scores, all_experiemnts, varying_params,
                       experiments_dir / 'brier_scores.png')
        _plot_and_save(plot_c_idxs_scores, all_experiemnts, varying_params,
                       experiments_dir / 'concordance_idxs.png')
        _plot_and_save(partial(plot_confusion_matrices, cm_type='cm_soft'),
                       all_experiemnts, varying_params,
                       experiments_dir / 'confusion_matrix_soft.png')

    _plot_and_save(plot_variance_mse, all_experiemnts, varying_params,
                   experiments_dir / 'variance_mse.png')


def _plot_and_save(plotting_func: Callable,
                   experiments: list[ExperimentRecord], varying_params: dict,
                   save_path: Path | str) -> None:

    save_path = Path(save_path)
    log.info(f'Plotting {save_path.name}')
    result = plotting_func(experiments, varying_params)
    if result == None:
        return
    figs = result if isinstance(result, tuple) else (result, )

    for i, fig in enumerate(figs):
        path = save_path if len(figs) == 1 else save_path.with_stem(
            f"{save_path.stem}_{i+1}")
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    experiments_path = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/start_frame_feature_comparison_20260120_152554'
    )

    # Plot main experiment results (accuracy, c-index, etc.)
    # plot_experiment_results(experiments_path, plot_type='box')

    # Plot confusion matrices
    # plot_confusion_matrices(experiments_path)

    # plot_experiment_losses(experiments_path, metric='total')
