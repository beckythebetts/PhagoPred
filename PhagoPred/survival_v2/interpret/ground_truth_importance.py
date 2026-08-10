from __future__ import annotations
from pathlib import Path
import os
import shutil

import torch
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.utils.io import load_model
from PhagoPred.survival_v2.configs.datasets import BinaryDatasetCfg, SurvivalDatasetCfg
from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ALL_CFGS, ScenarioCfg
from PhagoPred.survival_v2.data.graph_synthetic.analytic_estimates import sampleWithImportances, generate_sample_with_importances
from PhagoPred.survival_v2.interpret.SHAP_kernel import KernelSHAP, KernelSHAPResults
from PhagoPred.survival_v2.interpret.importance_plots import (
    feature_panel,
    heatmap_panel,
    outputs_panel,
    series_rows,
    temporal_panel,
)
from PhagoPred.survival_v2.interpret.importance_data import (
    dataset_average,
    horizon_outputs,
    load_sample_importances,
    read_root_attrs,
    sample_indices,
)

log = get_logger()

# h5py None
_NO_HAZARD_BINS = np.array([])


def _hazard_bins_attr(hazard_bins: None | np.ndarray) -> np.ndarray:
    return _NO_HAZARD_BINS if hazard_bins is None else hazard_bins


def compare_importance(
    experiment_dir: Path | str,
    num_samples: int = 100,
    min_horizon_cif: float = 0.05,
    num_permutations: int = 100,
    num_shap_samples: int | None = None,
    num_plot_samples: int = 10,
    num_background_samples: int = 8,
    num_segments: int = 50,
) -> None:
    """``num_segments`` sets the temporal resolution of *both* the ground-truth
    observational Shapley (segment-players) and the model KernelSHAP, so the two
    are computed on the same segmentation and are directly comparable. Both are
    clamped to the landmark frame count, so a large value gives the per-frame
    game on each side.
    """
    log.info(f'Starting compare importances on {experiment_dir}')
    experiment_dir = Path(experiment_dir)

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

    # === GET SCENARIO ===
    scenario = _infer_scenario(experiment_dir)
    if scenario is None:
        raise ValueError("No scaenario found")
    train_paths = ds_cfg.train_paths
    dataset_dir = Path(train_paths[0]).parent
    scenario.load(dataset_dir)
    log.info(f'Using scenario {scenario.filename}')

    # === GET SAMPLES ===
    samples_file = _get_samples(
        dataset_dir,
        scenario,
        num_samples,
        horizon,
        hazard_bins,
        num_permutations,
        min_horizon_cif,
        output_type,
        num_background_samples,
        num_segments,
    )
    results_file = experiment_dir / 'shap_samples.h5'
    shutil.copy(samples_file, results_file)
    log.info(f'Got ground truth SHAP samples')

    # === RUN Kernel SHAP and write to file ===
    # Model and ground truth share num_segments -> same segmentation, directly
    # comparable. analyse() clamps to the landmark frame count internally.
    # KernelSHAP background is real cells from the dataset (never the graph) — it
    # must be model-agnostic. Loaded once; sliced per sample below.
    background_pool = _load_background_pool(
        dataset_dir / f'{scenario.filename}_train.h5', model_feat_names,
        num_candidates=max(512, num_background_samples * 16))
    kernel_shap = KernelSHAP(model, model_feat_names, device)
    for sample_idx in range(num_samples):
        try:
            sample = sampleWithImportances.from_h5(results_file, sample_idx)
        except KeyError:
            log.warning(
                f'Sample {sample_idx} not found in {results_file}, skipping')
            continue
        model_input = _build_model_input(
            sample.base_signals,
            sample.landmark_frame,
            model_feat_names,
            means,
            stds,
        )
        # Distributional baseline from *real data*: masked features are replaced
        # by real cell trajectories and the prediction averaged, so the model's
        # completeness sum (pred - E[f]) matches the ground-truth Shapley scale —
        # rather than a mask-to-0 baseline, which gives f(mean) != E[f].
        value_background = _model_value_background(background_pool,
                                                   sample.landmark_frame, means,
                                                   stds, num_background_samples)
        shap_results = kernel_shap.analyse(
            model_input,
            torch.tensor([sample.landmark_frame]),
            num_segments,
            nsamples_temporal_feature=num_shap_samples,
            output_type=output_type,
            time_bins=hazard_bins,
            value_background=value_background)
        _write_model_shap_results(results_file, sample_idx, shap_results,
                                  model_feat_names)
        _write_model_prediction(
            results_file,
            sample_idx,
            _model_prediction(model, model_input, sample.landmark_frame,
                              output_type, device),
        )

    # === PLOT ===
    plot_samples(experiment_dir, num_plot_samples)
    average_fig = plot_dataset_average(experiment_dir)
    average_path = experiment_dir / 'ground_truth_comparison' / 'dataset_average.png'
    average_fig.savefig(average_path, bbox_inches='tight', dpi=120)
    plt.close(average_fig)
    log.info(f'Saved dataset average figure to {average_path}')


def _model_prediction(
    model,
    model_input: torch.Tensor,
    landmark_frame: int,
    output_type: str,
    device: str,
) -> np.ndarray:
    """The model's own prediction for this sample, to sit alongside the truth.

    Binary models emit a single P(event within horizon); survival models emit a
    per-bin PMF. Both are compared in plot_sample_on_axes against the quantity
    derived from the sample's stored ``horizon_hazard``.
    """
    with torch.no_grad():
        logits = model(model_input.to(device),
                       torch.tensor([landmark_frame]).to(device),
                       return_attention=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        pred = (model.predict_binary(logits)
                if output_type == 'binary' else model.predict_pmf(logits))
    return np.atleast_1d(pred.cpu().numpy().squeeze())


def _write_model_prediction(h5_file: Path, sample_idx: int,
                            prediction: np.ndarray) -> None:
    with h5py.File(h5_file, 'r+') as f:
        sample = f[str(sample_idx)]
        if 'Model Prediction' in sample:
            del sample['Model Prediction']
        sample.create_dataset('Model Prediction', data=prediction, dtype=float)


def _write_model_shap_results(h5_file: Path, sample_idx: int,
                              shap_results: KernelSHAPResults,
                              model_featurre_names: list[str]) -> None:
    with h5py.File(h5_file, 'r+') as f:
        sample = f[str(sample_idx)]
        model = sample.create_dataset(
            'Model',
            data=shap_results.temporal_feature_shap_values[0],
            dtype=float)
        model.attrs['Temporal'] = shap_results.temporal_shap_values[0]
        model.attrs['Feature'] = shap_results.feature_shap_values[0]
        model.attrs['Segment Boundaries'] = shap_results.segment_boundaries
        model.attrs['Feature Names'] = model_featurre_names


def _build_model_input(
    base_signals: dict[str, np.ndarray],
    landmark_frame: int,
    feature_names: list[str],
    means: np.ndarray | None,
    stds: np.ndarray | None,
) -> torch.Tensor:
    """Stack base_signals into a normalised (1, lf, num_feats) tensor for model input."""
    matrix = np.stack(
        [base_signals[f][:landmark_frame] for f in feature_names],
        axis=-1).astype(np.float32)
    if means is not None and stds is not None:
        safe_stds = np.where(stds == 0.0, 1.0, stds)
        matrix = (matrix - means) / safe_stds
    return torch.tensor(matrix).unsqueeze(0)


def _load_background_pool(dataset_path: Path,
                          feature_names: list[str],
                          num_candidates: int = 512,
                          seed: int = 0) -> np.ndarray:
    """A pool of candidate *real* cell trajectories — the model-agnostic KernelSHAP
    background.

    KernelSHAP knows nothing about the data-generating process, so the background
    must come from the data, not the causal graph. Whole cells are drawn (all
    features from one cell) to match the ground-truth interventional estimator,
    which averages over whole marginal rollouts. The dataset stores absent frames
    (late entry / death) as NaN, so more candidates than needed are loaded and
    ``_model_value_background`` keeps only those fully observed over [0, lf).
    Returns (num_candidates, T, F) raw signals (may contain NaN).
    """
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, 'r') as f:
        phase = f['Cells']['Phase']
        n_cells = phase[feature_names[0]].shape[1]  # (frames, cells)
        cells = rng.choice(n_cells,
                           size=min(num_candidates, n_cells),
                           replace=False)
        pool = np.stack([
            np.stack([phase[fn][:, c] for fn in feature_names], axis=-1)
            for c in cells
        ]).astype(np.float32)  # (num_candidates, T, F)
    return pool


def _model_value_background(pool: np.ndarray, landmark_frame: int,
                            means: np.ndarray | None, stds: np.ndarray | None,
                            num_draws: int) -> torch.Tensor:
    """Real-data background over [0, lf): keep fully-observed cells, normalise.

    A cell with any NaN in [0, lf) (late entry, or death within the window) would
    make the model output NaN, so those are dropped and up to ``num_draws``
    complete ones are kept. Returns (nbg, lf, F).
    """
    window = pool[:, :landmark_frame, :]  # (n_candidates, lf, F)
    observed = ~np.isnan(window).any(axis=(1, 2))  # cells present over [0, lf)
    valid = window[observed]
    if len(valid) == 0:
        raise ValueError(
            f'no fully-observed background cells over [0, {landmark_frame}); '
            'raise num_candidates in _load_background_pool')
    mat = valid[:num_draws].copy()
    if means is not None and stds is not None:
        safe_stds = np.where(stds == 0.0, 1.0, stds)
        mat = (mat - means) / safe_stds
    return torch.tensor(mat)


def _infer_scenario(experiment_dir: Path) -> ScenarioCfg | None:
    """Match the experiment's dataset paths to a known ScenarioCfg by filename stem."""
    with open(experiment_dir / 'config.json') as f:
        cfg_raw = json.load(f)
    dataset_cfg = cfg_raw.get('dataset', {})
    all_paths = dataset_cfg.get('train_paths', []) + dataset_cfg.get(
        'val_paths', [])
    for path in all_paths:
        stem = Path(path).stem  # '<scenario.filename>_<split>'
        for suffix in ('_train', '_val', '_cal'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        # Exact match: a substring test lets 'linear_chain_high_ar' capture the
        # dataset 'nonlinear_chain_high_ar', silently building GT from the wrong
        # graph. The dataset filename stem equals scenario.filename exactly.
        for scenario_cfg in ALL_CFGS:
            if scenario_cfg.filename == stem:
                return scenario_cfg
    return None


def _get_samples(
    daatset_dir: Path,
    scenario: ScenarioCfg,
    num_samples: int,
    horizon: int,
    hazard_bins: None | np.ndarray,
    num_permutations: int,
    min_horizon_cif: float,
    output_type: str,
    num_background_samples: int,
    num_segments: int | None = None,
):
    seg_attr = _segments_attr(num_segments)
    file_name = None
    samples_dir = daatset_dir / 'shap_samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    try:
        for file in samples_dir.iterdir():
            if scenario.filename in file.name:
                with h5py.File(file, 'r') as f:
                    if f.attrs['Output type'] == output_type:
                        if f.attrs['Horizon'] == horizon:
                            if np.array_equal(f.attrs['Hazard Bins'],
                                              _hazard_bins_attr(hazard_bins)):
                                if f.attrs[
                                        'Num Permutations'] == num_permutations:
                                    if f.attrs[
                                            'Min Horizon CIF'] == min_horizon_cif:
                                        if f.attrs[
                                                'Num Background Samples'] == num_background_samples:
                                            if f.attrs.get('Num Segments',
                                                           -1) == seg_attr:
                                                file_name = file
    except KeyError:
        pass

    if file_name is None:
        idx = 0
        file_name = samples_dir / f'{scenario.filename}_{idx}.h5'
        while file_name.is_file():
            idx += 1
            file_name = samples_dir / f'{scenario.filename}_{idx}.h5'
        _generate_samples(
            file_name,
            scenario,
            num_samples,
            horizon,
            hazard_bins,
            num_permutations,
            min_horizon_cif,
            output_type,
            num_background_samples,
            num_segments,
        )
    return file_name


def _segments_attr(num_segments: int | None) -> int:
    """h5-storable segment count; -1 encodes per-frame (None)."""
    return -1 if num_segments is None else int(num_segments)


def _generate_samples(
    h5_path: Path,
    scenario: ScenarioCfg,
    num_samples: int,
    horizon: int,
    hazard_bins: None | np.ndarray,
    num_permutations: int,
    min_horizon_cif: float,
    output_type: str,
    num_background_samples: int,
    num_segments: int | None = None,
) -> None:

    with h5py.File(h5_path, 'w') as f:
        f.attrs['Scenario'] = scenario.filename
        f.attrs['Output type'] = output_type
        f.attrs['Horizon'] = horizon
        f.attrs['Hazard Bins'] = _hazard_bins_attr(hazard_bins)
        f.attrs['Num Permutations'] = num_permutations
        f.attrs['Min Horizon CIF'] = min_horizon_cif
        f.attrs['Num Samples'] = num_samples
        f.attrs['Num Background Samples'] = num_background_samples
        f.attrs['Num Segments'] = _segments_attr(num_segments)

    attempt_budget = num_samples * 200
    attempts = 0
    samples = 0
    while samples < num_samples and attempts < attempt_budget:
        sample = generate_sample_with_importances(
            scenario.graph,
            scenario.hazard_calibration_func,
            horizon,
            scenario.num_frames,
            100,
            num_permutations,
            hazard_bins,
            output_type,
            min_horizon_cif,
            num_background_samples,
            num_segments=num_segments,
        )
        if sample is not None:
            sample.to_h5(h5_path, samples)
            samples += 1
            log.info(
                f'Generated sample {samples} / {num_samples}, {attempts} total attempts'
            )

        attempts += 1
    return samples


def _as_str(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def backfill_horizon_hazard(
    h5_path: Path,
    dataset_dir: Path,
    force: bool = False,
) -> int:
    """Add 'Horizon Hazard' to sample files written before that field existed.

    The calibrated hazard is a pure function of data already in the file — the
    Hazard row of 'Signals' over [lf, lf + horizon) — together with the
    scenario's calibration func, which lives in the dataset rather than in the
    sample file. So it is recoverable exactly, without regenerating the samples.

    ``_get_samples`` matches its cache on the root attrs only, so it will happily
    reuse an old file forever; this is the way to bring one up to date.

    The file is opened with HDF5 locking left on, so it will refuse to touch a
    file that another process is still writing.

    Returns the number of sample groups updated.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'r+') as f:
        scenario_name = _as_str(f.attrs['Scenario'])
        horizon = int(f.attrs['Horizon'])
        scenario = next((c for c in ALL_CFGS if c.filename == scenario_name),
                        None)
        if scenario is None:
            raise ValueError(
                f'Unknown scenario {scenario_name!r} in {h5_path}')
        scenario.load(Path(dataset_dir))

        # A file written with sample_idx=None keeps its datasets at the root.
        groups = [f] if 'Signals' in f else [f[k] for k in f]

        written = 0
        for group in groups:
            if 'Horizon Hazard' in group:
                if not force:
                    continue
                del group['Horizon Hazard']
            features = [_as_str(n) for n in group.attrs['Features']]
            hazard_row = features.index('Hazard')
            lf = int(group.attrs['Landmark Frame'])
            signal = group['Signals'][hazard_row, lf:lf + horizon]
            if len(signal) != horizon:
                log.warning(f'{h5_path.name}:{group.name} has only '
                            f'{len(signal)} frames past lf={lf}, expected '
                            f'{horizon}; skipping')
                continue
            group.create_dataset('Horizon Hazard',
                                 data=scenario.hazard_calibration_func(signal),
                                 dtype=float)
            written += 1
    return written


# ─────────────────────────── plotting ───────────────────────────
# Panel primitives live in importance_plots so experiments.plots can reuse them
# without importing torch. Per-sample panels are signed; dataset averages use
# mean |SHAP| (signed values from different samples cancel).


def plot_sample_on_axes(
    experiment_dir: Path,
    axes: list[plt.axes],
    sample_idx: int | None,
    signals: bool = True,
    interventional: bool = True,
    observational: bool = True,
    model: bool = True,
    temporal: bool = True,
    feature: bool = True,
    outputs: bool = True,
    normalise: bool = True,
) -> None:
    """Draw one stored sample across ``axes``, straight from shap_samples.h5.

    Panels are consumed in the order signals, interventional, observational,
    model, temporal, feature, outputs — one axis per enabled flag. The heatmaps
    share a [0, lf) frame axis so they can be read column-wise against each
    other; model SHAP is per-segment and is spread onto that frame grid.

    The temporal and feature panels collapse the model's joint (segment x
    feature) map. Ground-truth temporal curves are summed over the model's
    features only, so both sides sum over the same set; Hazard still appears in
    the heatmaps and the per-feature bars.

    ``sample_idx=None`` reads a single-sample file written at the h5 root.
    Panels whose data is absent are annotated rather than raising.
    """
    flags = [
        signals, interventional, observational, model, temporal, feature,
        outputs
    ]
    assert np.sum(flags) == len(axes)

    h5_path = Path(experiment_dir) / 'shap_samples.h5'
    root = read_root_attrs(h5_path)
    sample = load_sample_importances(h5_path, sample_idx)

    lf = sample.landmark_frame
    death_str = ('censored' if sample.death_frame is None else
                 f'{sample.death_frame:.0f}')
    remaining = list(axes)

    def _next() -> plt.Axes:
        return remaining.pop(0)

    def _norm(values):
        values = np.asarray(values, dtype=float)
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    def _missing(ax: plt.Axes, what: str) -> None:
        ax.text(0.5,
                0.5,
                f'No {what} in\n{h5_path.name}',
                ha='center',
                va='center',
                fontsize=8,
                transform=ax.transAxes)
        ax.set_axis_off()

    if signals:
        heatmap_panel(_next(),
                      sample.signals,
                      sample.feature_names,
                      lf,
                      f'Signals (row-normalised)  lf={lf}  death={death_str}',
                      row_normalise=True)

    for enabled, key in ((interventional, 'Interventional'),
                         (observational, 'Observational')):
        if not enabled:
            continue
        ax = _next()
        if key not in sample.ground_truth:
            _missing(ax, f'{key} importances')
        else:
            heatmap_panel(ax, sample.ground_truth[key], sample.feature_names,
                          lf, f'{key} GT SHAP')

    if model:
        ax = _next()
        if sample.model_map is None:
            _missing(ax, 'Model SHAP (run compare_importance)')
        else:
            n_segments = len(sample.segment_boundaries) - 1
            heatmap_panel(
                ax, sample.model_map, sample.model_feature_names, lf,
                f'Model KernelSHAP ({n_segments} segments, per-frame)')

    unit = ' (relative)' if normalise else ''
    if temporal:
        ax = _next()
        series = {
            k: _norm(sample.ground_truth_temporal(k))
            for k in sample.ground_truth
        }
        if sample.model_map is not None:
            series['Model'] = _norm(sample.model_temporal_from_map())
        temporal_panel(ax,
                       series, f'Temporal importance{unit} (summed over '
                       f'{", ".join(sample.shared_feature_names)})',
                       n_frames=lf)

    if feature:
        ax = _next()
        series = {
            k: (sample.feature_names, _norm(sample.ground_truth_feature(k)))
            for k in sample.ground_truth
        }
        if sample.model_map is not None:
            series['Model'] = (sample.model_feature_names,
                               _norm(sample.model_feature_from_map()))
        feature_panel(ax, series, f'Feature importance{unit} (separate game)')

    if outputs:
        outputs_panel(
            _next(),
            root['output_type'],
            ground_truth=horizon_outputs(sample.horizon_hazard,
                                         root['output_type'],
                                         root['hazard_bins']),
            model_prediction=sample.model_prediction,
            hazard_bins=root['hazard_bins'],
            horizon=root['horizon'],
            death_offset=(None if sample.death_frame is None else
                          sample.death_frame - lf),
        )


def plot_sample(experiment_dir: Path, sample_idx: int) -> plt.Figure:
    """Full seven-panel figure for one sample."""
    fig, axes = plt.subplots(1, 7, figsize=(36, 4))
    plot_sample_on_axes(experiment_dir, list(axes), sample_idx)
    fig.suptitle(f'Sample {sample_idx} — {Path(experiment_dir).name}',
                 fontweight='bold',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_samples(experiment_dir: Path, num_plot_samples: int) -> list[Path]:
    """Save one figure per sample into ``<experiment_dir>/ground_truth_comparison``."""
    experiment_dir = Path(experiment_dir)
    save_dir = experiment_dir / 'ground_truth_comparison'
    save_dir.mkdir(parents=True, exist_ok=True)

    indices = sample_indices(experiment_dir / 'shap_samples.h5')
    indices = indices[:num_plot_samples] if indices else [None]

    saved = []
    for sample_idx in indices:
        fig = plot_sample(experiment_dir, sample_idx)
        path = save_dir / f'sample_{sample_idx:02d}.png'
        fig.savefig(path, bbox_inches='tight', dpi=120)
        plt.close(fig)
        saved.append(path)
    log.info(f'Saved {len(saved)} sample figures to {save_dir}')
    return saved


def plot_dataset_average(experiment_dir: Path,
                         normalise: bool = True) -> plt.Figure:
    """Mean |SHAP| over every stored sample: heatmaps, then temporal and feature bars.

    Samples have different landmark frames, so maps are stacked on an absolute
    frame axis and NaN-padded; the shaded band on the temporal panel is how many
    samples reach each frame. The right-hand columns average over the handful of
    long-lf samples and are correspondingly noisy.

    ``normalise`` (default) rescales each estimator's values in each panel to sum
    to 1, so model and ground truth are compared on *relative* importance — the
    baseline choice (mask vs. distributional) and the model's larger output range
    otherwise leave the model 2-3x the ground-truth scale and visually dominant.
    """
    experiment_dir = Path(experiment_dir)
    average = dataset_average(experiment_dir / 'shap_samples.h5')

    def norm(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        total = np.nansum(np.abs(values))
        return values / total if (normalise and total > 0) else values

    unit = 'relative' if normalise else 'mean |SHAP|'
    map_keys = list(average.maps)
    fig, axes = plt.subplots(1,
                             len(map_keys) + 2,
                             figsize=(6 * (len(map_keys) + 2), 4))
    axes = list(axes)

    for key in map_keys:
        rows = series_rows(key, average.feature_names,
                           average.model_feature_names)
        heatmap_panel(axes.pop(0),
                      norm(average.maps[key]),
                      rows,
                      average.max_landmark_frame,
                      f'{key}  {unit}',
                      diverging=False)

    temporal_panel(axes.pop(0),
                   {k: norm(v) for k, v in average.temporal.items()},
                   f'Temporal importance  {unit}',
                   support=average.support)

    feature_series = {}
    for key, values in average.feature.items():
        rows = series_rows(key, average.feature_names,
                           average.model_feature_names)
        feature_series[key] = (rows, norm(values))
    feature_panel(axes.pop(0), feature_series, f'Feature importance  {unit}')

    fig.suptitle(
        f'Dataset average over {average.num_samples} samples — '
        f'{experiment_dir.name}',
        fontweight='bold',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig
