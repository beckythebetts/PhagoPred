"""Compare ground-truth permutation Shapley values against KernelSHAP on a trained model.

For each generated synthetic sample:
  - Computes ground-truth permutation Shapley values for each (feature, time-step) player
  - Runs KernelSHAP on the trained prediction model for the same sample input
  - Plots side-by-side heatmaps (and optionally CDF variance bounds)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import torch

from PhagoPred.survival_v2.data.graph_synthetic.analytic_estimates import (
    generate_sample_with_importances,
    sampleWithImportances,
)
from PhagoPred.survival_v2.configs.datasets import BinaryDatasetCfg
from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ALL_CFGS, ScenarioCfg
from PhagoPred.survival_v2.interpret.SHAP_kernel import KernelSHAP
from PhagoPred.survival_v2.utils.io import load_model


def _infer_scenario(experiment_dir: Path) -> ScenarioCfg | None:
    """Match the experiment's dataset paths to a known ScenarioCfg by filename stem."""
    with open(experiment_dir / 'config.json') as f:
        cfg_raw = json.load(f)
    dataset_cfg = cfg_raw.get('dataset', {})
    all_paths = dataset_cfg.get('train_paths', []) + dataset_cfg.get(
        'val_paths', [])
    for path in all_paths:
        stem = Path(path).stem
        for scenario_cfg in ALL_CFGS:
            if scenario_cfg.filename in stem:
                return scenario_cfg
    return None


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
    return torch.tensor(matrix).unsqueeze(0)  # (1, lf, num_feats)


def compare_importance(
    experiment_dir: Path,
    n_samples: int = 5,
    min_horizon_cif: float = 0.05,
    num_permutations: int = 50,
    permutations_per_batch: int = 5,
    output_type: Literal['expected_time', 'binary'] | None = None,
    horizon: int | None = None,
    hazard_bins: np.ndarray | None = None,
    num_segments: int = 30,
    nsamples_shap: int = 500,
    device: str | None = None,
    save_dir: Path | None = None,
    scenario: ScenarioCfg | None = None,
    show_variance_bounds: bool = False,
    variance_base_samples: int = 50,
    variance_branch_samples: int = 200,
    show_observational: bool = True,
    show_noise_observational: bool = True,
    show_interventional: bool = True,
    show_feature_values: bool = False,
    show_base_noise: bool = False,
    show_feature_totals: bool = True,
) -> list[sampleWithImportances]:
    """Generate synthetic samples, compute ground-truth + model SHAP, and plot a comparison.

    Parameters
    ----------
    experiment_dir
        Directory produced by the experiment suite (must contain config.json + model.pkl).
    n_samples
        Number of synthetic samples to generate and compare.
    min_horizon_cif
        Minimum in-horizon CIF (death probability within the horizon along the
        realised trajectory) a generated sample must have to be kept. Filters out
        degenerate censored-far-from-death cells whose target (RMST / binary CIF)
        is saturated, making every Shapley value ≈0. Set to 0.0 to disable.
    num_permutations
        Permutations per sample to estimate ground-truth Shapley values.
    permutations_per_batch
        How many permutations to batch into a single _apply_rules call (speed/memory trade-off).
    output_type
        Scalar output to explain — must match what the model outputs. If None,
        derived from the experiment's dataset config ('binary' for a binary
        dataset, otherwise 'expected_time').
    horizon
        Frames beyond the landmark used for hazard integration. If None, derived
        from the dataset config (prediction_horizon for binary, the final event
        time bin edge / max_time_to_death for survival).
    hazard_bins
        Bin edges for binned hazard; None uses per-frame hazard. If left None,
        the survival dataset's event time bins from the config are used (binary
        datasets remain unbinned).
    num_segments
        Temporal segments for KernelSHAP (fewer = faster, coarser).
    nsamples_shap
        KernelExplainer coalition samples for the temporal-feature analysis.
    device
        PyTorch device. If None, uses 'cuda' when available, else 'cpu'.
    save_dir
        Where to save figures; defaults to experiment_dir/ground_truth_comparison/.
    scenario
        ScenarioCfg to sample from; auto-inferred from config paths if None.
    show_variance_bounds
        If True, add a third column showing the model's predicted CDF alongside
        graph branching variance bounds (computationally expensive).
    variance_base_samples
        Base conditions sampled for branching variance (only used if show_variance_bounds).
    variance_branch_samples
        Branch trajectories per base condition for variance (only if show_variance_bounds).
    show_observational
        If True (default), add a column with the observational ground-truth SHAP
        (sample.observational_importances): past feature signals are pinned to
        their coalition values and the required innovations importance-weighted,
        so credit reflects the conditional (total causal) effect. This is the
        apples-to-apples target for a predictive model.
    show_noise_observational
        If True (default), add a column with the unweighted noise-swap
        observational baseline (sample.noise_observational_importances): feature
        innovations are perturbed and all graph rules re-applied with no pinning
        or reweighting. Kept alongside show_observational for comparison.
    show_interventional
        If True (default), add a column with the interventional (value-space)
        ground-truth SHAP: feature values are masked to the dataset mean and only
        the Hazard rule is re-applied, so credit reflects the direct effect of
        each feature on the hazard (the causal chain is severed). In chain
        scenarios this concentrates on the hazard's proximal parent only.
    show_feature_values
        If True, add a leftmost column showing the sample's true (raw, un-normalised)
        feature values as a heatmap over [0, lf), using the same feature×frame
        layout, frame axis and RdBu_r scaling as the SHAP panels so attributions
        line up row-for-row with the underlying signals.
    show_base_noise
        If True, add a column showing the sample's base noise (the per-feature
        innovations driving the signals, sample.base_noise) as a heatmap over
        [0, lf), using the same feature×frame layout, frame axis and RdBu_r
        scaling as the SHAP panels so attributions line up row-for-row with the
        underlying noise.
    show_feature_totals
        If True (default), add a column with a grouped horizontal bar chart of
        each feature's total importance (signed sum over all time steps) for the
        ground-truth and model SHAP attributions, so the two can be compared at
        the per-feature level independent of their temporal distribution.

    Returns
    -------
    list[sampleWithImportances]
        The generated samples; .sample_importances has shape
        (num_graph_feats, seg_count) where seg_count = min(num_segments, lf),
        computed over the same temporal segments as the model KernelSHAP.
    """
    experiment_dir = Path(experiment_dir)
    save_dir = Path(
        save_dir
    ) if save_dir is not None else experiment_dir / 'ground_truth_comparison'
    save_dir.mkdir(exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── load model ──────────────────────────────────────────────────────────
    model, cfg, checkpoint = load_model(experiment_dir, device=device)
    model_feat_names: list[str] = (cfg.feature_combo if isinstance(
        cfg.feature_combo, list) else [cfg.feature_combo])
    means = checkpoint.get('normalisation_means')
    stds = checkpoint.get('normalisation_stds')
    if means is not None:
        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)

    # ── resolve output_type / horizon / hazard_bins from the dataset config ──
    # These must match how the model was trained, so default them from the
    # experiment's dataset config unless explicitly overridden.
    ds_cfg = cfg.dataset
    is_binary = isinstance(ds_cfg, BinaryDatasetCfg)

    if output_type is None:
        output_type = 'binary' if is_binary else 'expected_time'

    if is_binary:
        if horizon is None:
            horizon = int(ds_cfg.prediction_horizon)
        # binary datasets are single-step CDF — no hazard binning
    else:
        bins = getattr(ds_cfg, 'bins', None)
        if hazard_bins is None and bins is not None:
            hazard_bins = np.asarray(bins)
        if horizon is None:
            horizon = int(bins[-1]) if bins is not None else int(
                ds_cfg.max_time_to_death)

    if horizon is None:
        raise ValueError(
            "Could not derive 'horizon' from the dataset config; pass horizon= explicitly."
        )
    print(
        f"Using output_type={output_type!r}, horizon={horizon}, "
        f"hazard_bins={'None' if hazard_bins is None else list(np.asarray(hazard_bins))}"
    )

    # ── resolve scenario ─────────────────────────────────────────────────────
    if scenario is None:
        scenario = _infer_scenario(experiment_dir)
    if scenario is None:
        raise ValueError(
            "Could not infer scenario from experiment config paths. "
            "Pass scenario= explicitly (a ScenarioCfg from scenarios.ALL_CFGS)."
        )
    # Load the calibration that was saved with the training data, so the
    # ground-truth hazard matches exactly what the model was trained on.
    # Re-running calibrate_hazard() would draw a fresh Monte-Carlo calibration
    # and silently diverge from the training distribution.
    if scenario.hazard_calibration_func is None:
        train_paths = getattr(cfg.dataset, 'train_paths', None)
        dataset_dir = Path(train_paths[0]).parent if train_paths else None
        calib_h5 = (dataset_dir / f'{scenario.filename}_train.h5'
                    if dataset_dir is not None else None)
        if calib_h5 is not None and calib_h5.exists():
            print(f"Loading saved hazard calibration from {calib_h5}...")
            scenario.load(dataset_dir)
        else:
            print(
                f"WARNING: training dataset for '{scenario.filename}' not found "
                f"at {calib_h5}; re-calibrating hazard (will NOT match the data "
                f"the model was trained on).")
            scenario.calibrate_hazard(
                target_death_fraction=scenario.target_death_fraction)

    hz = scenario.hazard_calibration_func
    graph = scenario.graph

    # Features shared between the graph and the model (ordered by model)
    graph_feat_names = [f.name for f in graph.features]
    model_feat_indices = [
        graph_feat_names.index(fn) for fn in model_feat_names
        if fn in graph_feat_names
    ]
    shared_feat_names = [graph_feat_names[i] for i in model_feat_indices]

    kernel_shap = KernelSHAP(model,
                             feature_names=shared_feat_names,
                             device=device)

    # For survival (expected_time) the model must explain the *restricted mean
    # survival time* (∫S dt over the bins) so it matches the ground-truth target,
    # which is RMST-based (outputs.expected_time = (survival·bin_widths).sum()).
    # Passing the bin left-edges routes _extract_output to
    # predict_restricted_survival_time; without them it falls back to
    # predict_expected_time — a truncated Σ pmf·bin_index with almost no dynamic
    # range, which collapses the model SHAP heatmap to ≈0. Length num_bins.
    model_time_bins = (np.asarray(hazard_bins)[:-1] if
                       (not is_binary and hazard_bins is not None) else None)

    # ── pre-compute variance bounds once (expensive) ─────────────────────────
    var_bounds: dict | None = None
    if show_variance_bounds and not is_binary:
        print("Computing graph branching variance bound")
        _, _, c_branch, c_total, _ = graph.get_variances(
            target_feature='Hazard',
            max_horizon=horizon,
            max_base_time_steps=scenario.num_frames,
            hazard_calibration_func=hz,
            base_sample_size=variance_base_samples,
            branch_sample_size=variance_branch_samples,
        )
        var_bounds = dict(
            c_branch=c_branch,  # aleatory: irreducible from fixed starting point
            c_total=c_total,  # total: aleatory + epistemic
        )

    # ── generate samples ─────────────────────────────────────────────────────
    # Filtering on in-horizon hazard rejects more draws, so allow a larger budget.
    attempt_budget = n_samples * (200 if min_horizon_cif > 0.0 else 20)
    samples: list[sampleWithImportances] = []
    attempts = 0
    while len(samples) < n_samples and attempts < attempt_budget:
        r = generate_sample_with_importances(
            graph=graph,
            hazard_calibration_func=hz,
            min_sequence_length=120,
            num_permutations=num_permutations,
            hazard_bins=hazard_bins,
            output_type=output_type,
            horizon=horizon,
            max_sequence_length=scenario.num_frames,
            min_horizon_cif=min_horizon_cif,
        )
        if r is not None:
            samples.append(r)
        attempts += 1

    if not samples:
        print('Not enough events found')
        return []
        # raise RuntimeError(
        #     f"Failed to generate any valid samples after {attempts} attempts "
        #     f"(min_horizon_cif={min_horizon_cif}). Lower min_horizon_cif if this "
        #     f"scenario rarely produces in-horizon events.")
    if len(samples) < n_samples:
        print(f"WARNING: only {len(samples)}/{n_samples} samples met "
              f"min_horizon_cif={min_horizon_cif} within {attempts} attempts.")

    # ── build figure ─────────────────────────────────────────────────────────
    # Column layout (left→right):
    #   [feature values] · [base noise] · GT SHAP · model SHAP · [totals] · [variance]
    # Need at least one ground-truth estimator to compare the model against.
    if not (show_observational or show_noise_observational
            or show_interventional):
        show_interventional = True

    _col = 0
    feat_col = None
    if show_feature_values:
        feat_col = _col
        _col += 1
    noise_col = None
    if show_base_noise:
        noise_col = _col
        _col += 1
    obs_col = None
    if show_observational:
        obs_col = _col
        _col += 1
    noise_obs_col = None
    if show_noise_observational:
        noise_obs_col = _col
        _col += 1
    interv_col = None
    if show_interventional:
        interv_col = _col
        _col += 1
    model_col = _col
    _col += 1
    totals_col = None
    if show_feature_totals:
        totals_col = _col
        _col += 1
    var_col = None
    if show_variance_bounds:
        var_col = _col
        _col += 1
    n_cols = _col

    for row_idx, sample in enumerate(samples):
        # One figure per sample (a single row of columns), saved separately.
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(6.5 * n_cols, 3.5),
            squeeze=False,
        )

        lf = int(sample.landmark_frame)
        death_str = 'censored' if sample.death_frame is None else str(
            int(sample.death_frame))

        # Slice both ground-truth estimators to model-shared features only.
        # Ground-truth is computed over the same temporal segments as the model,
        # so each is (num_shared, seg_count) — directly comparable to model_tf_shap.
        obs_imp = (sample.observational_importances[model_feat_indices, :]
                   if sample.observational_importances is not None else None)
        noise_obs_imp = (
            sample.noise_observational_importances[model_feat_indices, :]
            if sample.noise_observational_importances is not None else None)
        interv_src = (sample.interventional_importances
                      if sample.interventional_importances is not None else
                      sample.sample_importances)
        interv_imp = (interv_src[model_feat_indices, :]
                      if interv_src is not None else None)
        num_shared = len(shared_feat_names)

        seg_count = min(num_segments, lf)

        # Model input from base signals (normalised, no post-noise)
        x = _build_model_input(sample.base_signals, lf, shared_feat_names,
                               means, stds)

        print(
            f"\nSample {row_idx + 1}/{len(samples)}  lf={lf}  death={death_str}"
        )
        shap_results = kernel_shap.analyse(
            x=x,
            lengths=torch.tensor([lf]),
            num_segments=seg_count,
            nsamples_temporal=100,
            nsamples_temporal_feature=nsamples_shap,
            output_type=output_type,
            time_bins=model_time_bins,
            compute_temporal=False,
            compute_feature=False,
            compute_temporal_feature=True,
            show_progress=True,
        )

        # (num_segments, num_shared_feats) — signed SHAP values
        model_tf_shap = shap_results.temporal_feature_shap_values[0]

        # ── optional panel: true feature values heatmap ─────────────────────
        if show_feature_values:
            axf = axes[0, feat_col]
            # (num_shared, lf) — same feature×frame layout as the SHAP panels
            feat_matrix = np.stack([
                sample.base_signals[fname][:lf] for fname in shared_feat_names
            ],
                                   axis=0)
            vmaxf = max(float(np.abs(feat_matrix).max()), 1e-9)
            imf = axf.imshow(
                feat_matrix,
                aspect='auto',
                origin='lower',
                cmap='RdBu_r',
                vmin=-vmaxf,
                vmax=vmaxf,
                extent=[0, lf, -0.5, num_shared - 0.5],
            )
            axf.set_yticks(range(num_shared))
            axf.set_yticklabels(shared_feat_names, fontsize=8)
            axf.set_xlabel('frame', fontsize=8)
            axf.set_title(f'True feature values  lf={lf}', fontsize=8)
            fig.colorbar(imf, ax=axf, fraction=0.03, pad=0.02)

        # ── optional panel: base noise heatmap ───────────────────────────────
        if show_base_noise:
            axn = axes[0, noise_col]
            # (num_shared, lf) — same feature×frame layout as the SHAP panels
            noise_matrix = np.stack(
                [sample.base_noise[fname][:lf] for fname in shared_feat_names],
                axis=0)
            vmaxn = max(float(np.abs(noise_matrix).max()), 1e-9)
            imn = axn.imshow(
                noise_matrix,
                aspect='auto',
                origin='lower',
                cmap='RdBu_r',
                vmin=-vmaxn,
                vmax=vmaxn,
                extent=[0, lf, -0.5, num_shared - 0.5],
            )
            axn.set_yticks(range(num_shared))
            axn.set_yticklabels(shared_feat_names, fontsize=8)
            axn.set_xlabel('frame', fontsize=8)
            axn.set_title(f'Base noise  lf={lf}', fontsize=8)
            fig.colorbar(imn, ax=axn, fraction=0.03, pad=0.02)

        # ── ground-truth SHAP heatmap(s) ─────────────────────────────────────
        def _gt_panel(ax, imp, label):
            vmax = max(float(np.abs(imp).max()), 1e-9)
            im = ax.imshow(
                imp,
                aspect='auto',
                origin='lower',
                cmap='RdBu_r',
                vmin=-vmax,
                vmax=vmax,
                extent=[0, lf, -0.5, num_shared - 0.5],
            )
            ax.set_yticks(range(num_shared))
            ax.set_yticklabels(shared_feat_names, fontsize=8)
            ax.set_xlabel('frame', fontsize=8)
            ax.set_title(
                f"{label} GT SHAP  lf={lf}  death={death_str}\n"
                f"({num_permutations} perms, {seg_count} segs)",
                fontsize=8,
            )
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        if obs_col is not None and obs_imp is not None:
            _gt_panel(axes[0, obs_col], obs_imp,
                      'Observational (pinned/reweighted)')
        if noise_obs_col is not None and noise_obs_imp is not None:
            _gt_panel(axes[0, noise_obs_col], noise_obs_imp,
                      'Observational (noise-swap)')
        if interv_col is not None and interv_imp is not None:
            _gt_panel(axes[0, interv_col], interv_imp,
                      'Interventional (value-space)')

        # ── middle panel: model KernelSHAP temporal-feature ─────────────────
        ax1 = axes[0, model_col]
        vmax1 = max(float(np.abs(model_tf_shap).max()), 1e-9)
        im1 = ax1.imshow(
            model_tf_shap.T,  # → (num_shared, num_segments)
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax1,
            vmax=vmax1,
            extent=[0, lf, -0.5, num_shared - 0.5],
        )
        ax1.set_yticks(range(num_shared))
        ax1.set_yticklabels(shared_feat_names, fontsize=8)
        ax1.set_xlabel('frame', fontsize=8)
        baseline = shap_results.temporal_feature_baseline
        ax1.set_title(
            f"Model KernelSHAP  ({seg_count} segs, baseline={baseline:.3f})\n"
            f"({nsamples_shap} coalition samples, mask=0)",
            fontsize=8,
        )
        fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)

        # ── panel: per-feature total importance (Σ over frames) ──────────────
        if show_feature_totals:
            axt = axes[0, totals_col]
            # One grouped bar per enabled estimator + the model (Σ over frames).
            series = []
            if obs_col is not None and obs_imp is not None:
                series.append(('Observational', obs_imp.sum(axis=1), 'C0'))
            if noise_obs_col is not None and noise_obs_imp is not None:
                series.append(('Noise obs.', noise_obs_imp.sum(axis=1), 'C3'))
            if interv_col is not None and interv_imp is not None:
                series.append(('Interventional', interv_imp.sum(axis=1), 'C2'))
            series.append(('Model', model_tf_shap.sum(axis=0), 'C1'))

            y = np.arange(num_shared)
            n_series = len(series)
            bh = 0.8 / n_series
            for i, (label, vals, color) in enumerate(series):
                offset = (i - (n_series - 1) / 2) * bh
                axt.barh(y + offset, vals, height=bh, color=color, label=label)
            axt.axvline(0, color='grey', lw=0.5)
            axt.set_yticks(y)
            axt.set_yticklabels(shared_feat_names, fontsize=8)
            axt.set_ylim(-0.5, num_shared - 0.5)  # match heatmap row layout
            axt.set_xlabel('total importance (Σ over frames)', fontsize=8)
            axt.set_title('Per-feature total importance', fontsize=8)
            axt.legend(fontsize=7)

        # ── right panel: model prediction vs graph uncertainty ──────────────
        if show_variance_bounds:
            ax2 = axes[0, var_col]
            with torch.no_grad():
                logits = model(x.to(device),
                               torch.tensor([lf]).to(device),
                               return_attention=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # Binary model (num_bins=1) → predict_binary (sigmoid of the
                # calibrated logit); survival model → per-bin CDF.
                pred = (model.predict_binary(logits)
                        if is_binary else model.predict_cif(logits))
                pred = np.atleast_1d(pred.cpu().numpy().squeeze())

            if is_binary:
                # Binary model outputs a single P(event within horizon). Compare
                # it against the per-sample aleatory distribution of the true
                # CIF(H): branch many stochastic futures from this cell's state
                # at lf. The spread is the irreducible noise floor and the
                # model's prediction should sit near the distribution's mean.
                p_death = float(pred[-1])
                cif_dist = graph.branch_cif_at_horizon(
                    base_signals=sample.base_signals,
                    landmark_frame=lf,
                    max_horizon=horizon,
                    hazard_calibration_func=hz,
                    target_feature='Hazard',
                    branch_sample_size=variance_branch_samples,
                )
                ax2.hist(
                    cif_dist,
                    bins=30,
                    range=(0, 1),
                    density=True,
                    color='grey',
                    alpha=0.5,
                    label=f'Ground-truth CIF(H)\n(aleatory, n={len(cif_dist)})'
                )
                ax2.axvline(cif_dist.mean(),
                            color='orange',
                            lw=1.5,
                            label=f'GT mean={cif_dist.mean():.3f}')
                ax2.axvline(p_death,
                            color='steelblue',
                            lw=1.5,
                            label=f'Model P(death)={p_death:.3f}')
                ax2.set_xlim(0, 1)
                ax2.set_xlabel(f'CIF at horizon (t+{horizon})', fontsize=8)
                ax2.set_ylabel('density', fontsize=8)
                ax2.set_title('Model P(death) vs aleatory CIF(H) distribution',
                              fontsize=8)
                ax2.legend(fontsize=7)
            else:
                n_t = len(pred)
                ax2.plot(np.arange(n_t),
                         pred,
                         color='steelblue',
                         lw=1.5,
                         label='Model CDF')

                if var_bounds is not None:
                    h = min(n_t, len(var_bounds['c_total']))
                    ta = np.arange(h)
                    # Total variance band (outer): epistemic + aleatory
                    std_total = np.sqrt(
                        np.clip(var_bounds['c_total'][:h], 0, None))
                    ax2.fill_between(ta,
                                     np.clip(0.5 - std_total, 0, 1),
                                     np.clip(0.5 + std_total, 0, 1),
                                     alpha=0.15,
                                     color='grey',
                                     label='Total var ±1σ')
                    # Branch variance band (inner): aleatory only
                    std_branch = np.sqrt(
                        np.clip(var_bounds['c_branch'][:h], 0, None))
                    ax2.fill_between(ta,
                                     np.clip(0.5 - std_branch, 0, 1),
                                     np.clip(0.5 + std_branch, 0, 1),
                                     alpha=0.25,
                                     color='orange',
                                     label='Aleatory var ±1σ')

                if sample.death_frame is not None:
                    rel = sample.death_frame - lf
                    if 0 <= rel < n_t:
                        ax2.axvline(rel,
                                    color='red',
                                    ls='--',
                                    lw=1,
                                    label=f'death @ t+{int(rel)}')

                ax2.set_ylim(0, 1)
                ax2.set_xlabel('horizon (frames past lf)', fontsize=8)
                ax2.set_ylabel('CDF', fontsize=8)
                ax2.set_title('Model CDF vs graph variance bounds', fontsize=8)
                ax2.legend(fontsize=7)

        fig.suptitle(
            f"Ground-truth vs Model SHAP — '{scenario.filename}'  "
            f"({output_type})  lf={lf}  death={death_str}",
            fontweight='bold',
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = (save_dir /
                    f'ground_truth_vs_model_shap_sample_{row_idx:02d}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}")

    return samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare ground-truth permutation SHAP vs model KernelSHAP'
    )
    parser.add_argument('experiment_dir',
                        type=Path,
                        help='Experiment directory (config.json + model.pkl)')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--num_permutations', type=int, default=50)
    parser.add_argument('--permutations_per_batch', type=int, default=5)
    parser.add_argument(
        '--output_type',
        default=None,
        choices=['expected_time', 'binary'],
        help='Defaults to the dataset type in the experiment config.')
    parser.add_argument('--horizon',
                        type=int,
                        default=None,
                        help='Defaults to the prediction horizon / event-time '
                        'bins in the experiment config.')
    parser.add_argument('--num_segments', type=int, default=30)
    parser.add_argument('--nsamples_shap', type=int, default=500)
    parser.add_argument('--device',
                        default=None,
                        help="PyTorch device; defaults to cuda if available.")
    parser.add_argument('--save_dir', type=Path, default=None)
    parser.add_argument('--show_variance_bounds', action='store_true')
    parser.add_argument('--show_feature_values', action='store_true')
    parser.add_argument('--show_base_noise', action='store_true')
    parser.add_argument(
        '--show_feature_totals',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Per-feature total (Σ over frames) importance bar '
        'chart comparing GT vs model. On by default.')
    parser.add_argument('--variance_base_samples', type=int, default=50)
    parser.add_argument('--variance_branch_samples', type=int, default=200)
    parser.add_argument(
        '--scenario',
        default=None,
        help='Scenario filename stem, e.g. base_chain. Inferred if omitted.')
    args = parser.parse_args()

    sc = None
    if args.scenario is not None:
        sc = next((c for c in ALL_CFGS if c.filename == args.scenario), None)
        if sc is None:
            known = [c.filename for c in ALL_CFGS]
            raise ValueError(
                f"Unknown scenario '{args.scenario}'. Known: {known}")

    compare_importance(
        experiment_dir=args.experiment_dir,
        n_samples=args.n_samples,
        num_permutations=args.num_permutations,
        permutations_per_batch=args.permutations_per_batch,
        output_type=args.output_type,
        horizon=args.horizon,
        num_segments=args.num_segments,
        nsamples_shap=args.nsamples_shap,
        device=args.device,
        save_dir=args.save_dir,
        show_variance_bounds=args.show_variance_bounds,
        show_feature_values=args.show_feature_values,
        show_base_noise=args.show_base_noise,
        show_feature_totals=args.show_feature_totals,
        variance_base_samples=args.variance_base_samples,
        variance_branch_samples=args.variance_branch_samples,
        scenario=sc,
    )
