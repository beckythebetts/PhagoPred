from __future__ import annotations
from pathlib import Path
import json
from typing import Literal, Callable
from dataclasses import asdict, fields
from functools import partial

import matplotlib.pyplot as plt

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

log = get_logger()

plt.rcParams['font.family'] = 'serif'


def plot_experiment_results(experiments_dir: Path) -> None:
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

            all_experiemnts.append(
                ExperimentRecord(config, results, training_history))
            log.info(f'Gathered resulsts for experiment {asdict(config)}')
            for f in fields(ExperimentCfg):
                val = getattr(config, f.name)
                if val not in varying_params[f.name]:
                    varying_params[f.name].append(val)

    varying_params = {k: v for k, v in varying_params.items() if len(v) > 1}
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
    # if len(varying_params) == 0:
    #     print('No varying configurations found, skipping plotting')

    # if len(varying_params) == 1:
    #     fig = plot_box_plots_1var(all_experiemnts, varying_params)
    #     log.info('Saving results figure')
    #     plt.savefig(experiments_dir / 'results.png',
    #                 bbox_inches='tight',
    #                 dpi=150)
    #     plt.close(fig)

    #     fig = plt_cms_1var(all_experiemnts, varying_params)
    #     log.info('Saving confusion matrices figure')
    #     plt.savefig(experiments_dir / 'confusion_matrcies.png',
    #                 bbox_inches='tight',
    #                 dpi=150)
    #     plt.close(fig)

    #     fig = plot_experiment_losses_1var(all_experiemnts, varying_params)
    #     log.info('Saving all losses plot')
    #     plt.savefig(experiments_dir / 'losses.png',
    #                 bbox_inches='tight',
    #                 dpi=150)
    #     plt.close(fig)

    #     if is_binary:
    #         fig = plot_rocs_1var(all_experiemnts, varying_params)
    #         log.info('Saving ROC plot')
    #         plt.savefig(experiments_dir / 'roc.png')
    #         plt.close(fig)

    # elif len(varying_params) == 2:
    #     fig = plot_box_plots_2var(all_experiemnts, varying_params)
    #     log.info('Saving results figure')
    #     plt.savefig(experiments_dir / 'results.png',
    #                 bbox_inches='tight',
    #                 dpi=150)
    #     plt.close(fig)

    #     fig = plot_cms_2var(all_experiemnts, varying_params)
    #     log.info('Saving confusion matrices figure')
    #     plt.savefig(experiments_dir / 'confusion_matrcies.png',
    #                 bbox_inches='tight',
    #                 dpi=150)
    #     plt.close(fig)
    # else:
    #     print(
    #         f'More than 2 varying parameters {list(varying_params.keys())}, skipping plot'
    #     )


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
    plot_experiment_results(experiments_path, plot_type='box')

    # Plot confusion matrices
    # plot_confusion_matrices(experiments_path)

    # plot_experiment_losses(experiments_path, metric='total')
