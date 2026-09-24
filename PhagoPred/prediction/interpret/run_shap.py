from __future__ import annotations
from pathlib import Path
import shutil

import torch
import numpy as np
import h5py
from tqdm import tqdm

from PhagoPred.utils.logger import get_logger
from PhagoPred.prediction.utils.io import load_model
from PhagoPred.prediction.configs.datasets import BinaryDatasetCfg, SurvivalDatasetCfg
from PhagoPred.prediction.interpret.ground_truth import infer_scenario, gt_get_samples
from PhagoPred.prediction.interpret.model import get_samples as model_get_samples
from PhagoPred.prediction.interpret.model.utils import (
    load_background_pool, background_rows_for_length)
from PhagoPred.prediction.interpret.model.var_precision import fit_var
from PhagoPred.prediction.interpret.model.kernel_shap import (
    KernelSHAP, InterventionalBackground, ObservationalBackground,
    analyse_sample_in_file)
from PhagoPred.prediction.interpret import plots

log = get_logger()


def run_shap(
    experiment_dir: Path,
    num_samples: int = 100,
    num_shap_samples: int | None = None,
    num_time_segments: int = 50,
    num_permutations: int = 100,
    num_background_samples: int = 16,
    min_horizon_cif: int = 0.05,
) -> None:
    experiment_dir = Path(experiment_dir)
    log.info(f'Starting SHAP on {experiment_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Using device {device}')

    # === GET MODEL ===
    model, cfg, checkpoint = load_model(experiment_dir, device)
    model_feat_names: list[str] = (cfg.feature_combo if isinstance(
        cfg.feature_combo, list) else [cfg.feature_combo])
    means = checkpoint.get('normalisation_means')
    stds = checkpoint.get('normalisation_stds')
    if means is not None:
        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)

    # === GET DATASET CFG ===
    ds_cfg = cfg.dataset
    if isinstance(ds_cfg, BinaryDatasetCfg):
        output_type = 'binary'
        horizon = int(ds_cfg.prediction_horizon)
        hazard_bins = None
    elif isinstance(ds_cfg, SurvivalDatasetCfg):
        output_type = 'expected_time'
        horizon = ds_cfg.bins[-1]
        hazard_bins = np.array(ds_cfg.bins)
    else:
        raise TypeError("Datset cfg must be Binary or Survival")
    log.info(
        f'Output type: {output_type}, horizon: {horizon}, hazard_bins: {hazard_bins}'
    )

    # === GET SCENARIO ====
    scenario = infer_scenario(experiment_dir)
    if scenario is not None:
        train_paths = ds_cfg.train_paths
        dataset_dir = Path(train_paths[0]).parent
        scenario.load(dataset_dir)
        log.info(f'Using scenario {scenario.filename}')
    else:
        log.info(f'No scenario found, no ground truth SHAP will be evaluated.')

    # === GET SAMPLES ===
    if scenario is not None:
        samples_file = gt_get_samples(
            dataset_dir,
            scenario,
            num_samples,
            horizon,
            hazard_bins,
            num_permutations,
            min_horizon_cif,
            output_type=output_type,
            num_background_samples=num_background_samples,
            num_segments=num_time_segments)

        results_file = experiment_dir / 'shap_samples.h5'
        shutil.copy(samples_file, results_file)
        log.info(f'Got ground truth SHAP samples')

    else:
        results_file = model_get_samples(
            experiment_dir,
            num_samples,
        )

    # === RUN KERNEL SHAP ===
    pool, _ = load_background_pool(results_file, model_feat_names)
    var_fit = fit_var(pool, model_feat_names)
    log.info(
        f'Fit global VAR(p={var_fit.p}) background model over '
        f'{pool.shape[0]} samples ({"ground truth" if scenario else "real"} '
        f'background pool)')

    kernel_shap = KernelSHAP(model, model_feat_names, device)
    observational = ObservationalBackground(
        var_fit, num_cond_samples=num_background_samples)

    analyse_kwargs = dict(
        num_segments=num_time_segments,
        # None (the default) lets KernelSHAP.analyse auto-scale each axis's
        # nsamples to its own player count; an explicit num_shap_samples
        # overrides all three axes to that one flat value instead.
        nsamples_temporal=num_shap_samples,
        nsamples_feature=num_shap_samples,
        output_type=output_type,
        time_bins=hazard_bins,
        show_progress=False,
    )

    with h5py.File(results_file, 'a') as f:
        indices = sorted(int(k) for k in f.keys() if k.isdigit())
        for i in tqdm(indices,
                      desc='KernelSHAP (interventional + observational)'):
            sample_len = min(int(f[str(i)].attrs['Landmark Frame']),
                             f[str(i)]['Signals'].shape[1])
            interventional = InterventionalBackground(
                value_background=torch.tensor(background_rows_for_length(
                    pool, sample_len, num_rows=num_background_samples),
                                              dtype=torch.float32,
                                              device=device))
            for background in (interventional, observational):
                analyse_sample_in_file(f,
                                       i,
                                       kernel_shap,
                                       background,
                                       model_feat_names=model_feat_names,
                                       device=device,
                                       **analyse_kwargs)
    log.info(f'Wrote model KernelSHAP results to {results_file}')


def plot_shap(experiment_dir: Path | str) -> None:
    """Plot and save shap results of experiment"""
    print(f'Plotting SHAp results for experiment: {experiment_dir}')
    log.info(f'Plotting SHAp results for experiment: {experiment_dir}')
    experiment_dir = Path(experiment_dir)
    save_dir = experiment_dir / 'shap'
    save_dir.mkdir(parents=True, exist_ok=True)
    h5_path = experiment_dir / 'shap_samples.h5'

    av_fig = plots.plot_dataset_average(h5_path, '', normalise=False)
    av_fig.savefig(save_dir / 'dataset_average.png',
                   bbox_inches='tight',
                   dpi=120)

    plots.plot_samples(h5_path, save_dir, 10)
