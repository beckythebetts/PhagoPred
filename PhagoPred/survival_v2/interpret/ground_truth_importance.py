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
    generate_sample_with_feature_importance_batched,
    sampleWithImportances,
)
from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ALL_CFGS, ScenarioCfg
from PhagoPred.survival_v2.interpret.SHAP_kernel import KernelSHAP
from PhagoPred.survival_v2.utils.io import load_model


def _infer_scenario(experiment_dir: Path) -> ScenarioCfg | None:
    """Match the experiment's dataset paths to a known ScenarioCfg by filename stem."""
    with open(experiment_dir / 'config.json') as f:
        cfg_raw = json.load(f)
    dataset_cfg = cfg_raw.get('dataset', {})
    all_paths = dataset_cfg.get('train_paths', []) + dataset_cfg.get('val_paths', [])
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
        [base_signals[f][:landmark_frame] for f in feature_names], axis=-1
    ).astype(np.float32)
    if means is not None and stds is not None:
        safe_stds = np.where(stds == 0.0, 1.0, stds)
        matrix = (matrix - means) / safe_stds
    return torch.tensor(matrix).unsqueeze(0)  # (1, lf, num_feats)



def compare_importance(
    experiment_dir: Path,
    n_samples: int = 5,
    num_permutations: int = 50,
    permutations_per_batch: int = 5,
    output_type: Literal['expected_time', 'binary'] = 'expected_time',
    horizon: int = 100,
    hazard_bins: np.ndarray | None = None,
    num_segments: int = 30,
    nsamples_shap: int = 500,
    device: str = 'cpu',
    save_dir: Path | None = None,
    scenario: ScenarioCfg | None = None,
    show_variance_bounds: bool = False,
    variance_base_samples: int = 50,
    variance_branch_samples: int = 200,
) -> list[sampleWithImportances]:
    """Generate synthetic samples, compute ground-truth + model SHAP, and plot a comparison.

    Parameters
    ----------
    experiment_dir
        Directory produced by the experiment suite (must contain config.json + model.pkl).
    n_samples
        Number of synthetic samples to generate and compare.
    num_permutations
        Permutations per sample to estimate ground-truth Shapley values.
    permutations_per_batch
        How many permutations to batch into a single _apply_rules call (speed/memory trade-off).
    output_type
        Scalar output to explain — must match what the model outputs.
    horizon
        Frames beyond the landmark used for hazard integration.
    hazard_bins
        Bin edges for binned hazard; None uses per-frame hazard.
    num_segments
        Temporal segments for KernelSHAP (fewer = faster, coarser).
    nsamples_shap
        KernelExplainer coalition samples for the temporal-feature analysis.
    device
        PyTorch device.
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

    Returns
    -------
    list[sampleWithImportances]
        The generated samples; .sample_importances has shape (num_graph_feats, lf).
    """
    experiment_dir = Path(experiment_dir)
    save_dir = Path(save_dir) if save_dir is not None else experiment_dir / 'ground_truth_comparison'
    save_dir.mkdir(exist_ok=True)

    # ── load model ──────────────────────────────────────────────────────────
    model, cfg, checkpoint = load_model(experiment_dir, device=device)
    model_feat_names: list[str] = (
        cfg.feature_combo if isinstance(cfg.feature_combo, list)
        else [cfg.feature_combo]
    )
    means = checkpoint.get('normalization_means')
    stds = checkpoint.get('normalization_stds')
    if means is not None:
        means = np.array(means, dtype=np.float32)
        stds = np.array(stds, dtype=np.float32)

    # ── resolve scenario ─────────────────────────────────────────────────────
    if scenario is None:
        scenario = _infer_scenario(experiment_dir)
    if scenario is None:
        raise ValueError(
            "Could not infer scenario from experiment config paths. "
            "Pass scenario= explicitly (a ScenarioCfg from scenarios.ALL_CFGS)."
        )
    if scenario.hazard_calibration_func is None:
        print(f"Calibrating hazard for '{scenario.filename}'...")
        scenario.calibrate_hazard(target_death_fraction=scenario.target_death_fraction)

    hz = scenario.hazard_calibration_func
    graph = scenario.graph

    # Features shared between the graph and the model (ordered by model)
    graph_feat_names = [f.name for f in graph.features]
    model_feat_indices = [
        graph_feat_names.index(fn) for fn in model_feat_names if fn in graph_feat_names
    ]
    shared_feat_names = [graph_feat_names[i] for i in model_feat_indices]

    kernel_shap = KernelSHAP(model, feature_names=shared_feat_names, device=device)

    # ── pre-compute variance bounds once (expensive) ─────────────────────────
    var_bounds: dict | None = None
    if show_variance_bounds:
        print("Computing graph branching variance bounds (this may take a minute)...")
        _, _, c_branch, c_total, _ = graph.get_variances(
            target_feature='Hazard',
            max_horizon=horizon,
            max_base_time_steps=scenario.num_frames,
            hazard_calibration_func=hz,
            base_sample_size=variance_base_samples,
            branch_sample_size=variance_branch_samples,
        )
        var_bounds = dict(
            c_branch=c_branch,   # aleatory: irreducible from fixed starting point
            c_total=c_total,     # total: aleatory + epistemic
        )

    # ── generate samples ─────────────────────────────────────────────────────
    samples: list[sampleWithImportances] = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 20:
        r = generate_sample_with_feature_importance_batched(
            graph=graph,
            hazard_calibration_func=hz,
            horizon=horizon,
            max_sequence_length=scenario.num_frames,
            min_sequence_length=120,
            num_permutations=num_permutations,
            hazard_bins=hazard_bins,
            output_type=output_type,
            permutations_per_batch=permutations_per_batch,
        )
        if r is not None:
            samples.append(r)
        attempts += 1

    if not samples:
        raise RuntimeError(f"Failed to generate any valid samples after {attempts} attempts.")

    # ── build figure ─────────────────────────────────────────────────────────
    n_cols = 3 if show_variance_bounds else 2
    fig, axes = plt.subplots(
        len(samples), n_cols,
        figsize=(6.5 * n_cols, 3.5 * len(samples)),
        squeeze=False,
    )

    for row_idx, sample in enumerate(samples):
        lf = int(sample.landmark_frame)
        death_str = 'censored' if sample.death_frame is None else str(int(sample.death_frame))

        # Slice ground-truth importances to model-shared features only
        gt_imp = sample.sample_importances[model_feat_indices, :]  # (num_shared, lf)
        num_shared = len(shared_feat_names)

        seg_count = min(num_segments, lf)

        # Model input from base signals (normalised, no post-noise)
        x = _build_model_input(sample.base_signals, lf, shared_feat_names, means, stds)

        print(f"\nSample {row_idx + 1}/{len(samples)}  lf={lf}  death={death_str}")
        shap_results = kernel_shap.analyse(
            x=x,
            lengths=torch.tensor([lf]),
            num_segments=seg_count,
            nsamples_temporal=100,
            nsamples_temporal_feature=nsamples_shap,
            output_type=output_type,
            compute_temporal=False,
            compute_feature=False,
            compute_temporal_feature=True,
            show_progress=True,
        )

        # (num_segments, num_shared_feats) — signed SHAP values
        model_tf_shap = shap_results.temporal_feature_shap_values[0]

        # ── left panel: ground-truth SHAP heatmap ───────────────────────────
        ax0 = axes[row_idx, 0]
        vmax0 = max(float(np.abs(gt_imp).max()), 1e-9)
        im0 = ax0.imshow(
            gt_imp,
            aspect='auto', origin='lower', cmap='RdBu_r',
            vmin=-vmax0, vmax=vmax0,
            extent=[0, lf, -0.5, num_shared - 0.5],
        )
        ax0.set_yticks(range(num_shared))
        ax0.set_yticklabels(shared_feat_names, fontsize=8)
        ax0.set_xlabel('frame', fontsize=8)
        ax0.set_title(
            f"Ground-truth SHAP  lf={lf}  death={death_str}\n"
            f"({num_permutations} permutations, noise-space)",
            fontsize=8,
        )
        fig.colorbar(im0, ax=ax0, fraction=0.03, pad=0.02)

        # ── middle panel: model KernelSHAP temporal-feature ─────────────────
        ax1 = axes[row_idx, 1]
        vmax1 = max(float(np.abs(model_tf_shap).max()), 1e-9)
        im1 = ax1.imshow(
            model_tf_shap.T,            # → (num_shared, num_segments)
            aspect='auto', origin='lower', cmap='RdBu_r',
            vmin=-vmax1, vmax=vmax1,
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

        # ── right panel: model CDF vs variance bounds ────────────────────────
        if show_variance_bounds:
            ax2 = axes[row_idx, 2]
            with torch.no_grad():
                logits = model(x.to(device), torch.tensor([lf]).to(device),
                               return_attention=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred_cdf = model.predict_cif(logits).cpu().numpy().squeeze()

            t_axis = np.arange(len(pred_cdf))
            ax2.plot(t_axis, pred_cdf, color='steelblue', lw=1.5, label='Model CDF')

            if var_bounds is not None:
                h = min(len(pred_cdf), len(var_bounds['c_total']))
                ta = np.arange(h)
                # Total variance band (outer): epistemic + aleatory
                std_total = np.sqrt(np.clip(var_bounds['c_total'][:h], 0, None))
                ax2.fill_between(ta, np.clip(0.5 - std_total, 0, 1),
                                 np.clip(0.5 + std_total, 0, 1),
                                 alpha=0.15, color='grey', label='Total var ±1σ')
                # Branch variance band (inner): aleatory only
                std_branch = np.sqrt(np.clip(var_bounds['c_branch'][:h], 0, None))
                ax2.fill_between(ta, np.clip(0.5 - std_branch, 0, 1),
                                 np.clip(0.5 + std_branch, 0, 1),
                                 alpha=0.25, color='orange', label='Aleatory var ±1σ')

            if sample.death_frame is not None:
                rel = sample.death_frame - lf
                if 0 <= rel < len(pred_cdf):
                    ax2.axvline(rel, color='red', ls='--', lw=1,
                                label=f'death @ t+{int(rel)}')

            ax2.set_ylim(0, 1)
            ax2.set_xlabel('horizon (frames past lf)', fontsize=8)
            ax2.set_ylabel('CDF', fontsize=8)
            ax2.set_title('Model CDF vs graph variance bounds', fontsize=8)
            ax2.legend(fontsize=7)

    fig.suptitle(
        f"Ground-truth vs Model SHAP — '{scenario.filename}'  ({output_type})",
        fontweight='bold', fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = save_dir / 'ground_truth_vs_model_shap.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out_path}")
    return samples


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare ground-truth permutation SHAP vs model KernelSHAP'
    )
    parser.add_argument('experiment_dir', type=Path,
                        help='Experiment directory (config.json + model.pkl)')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--num_permutations', type=int, default=50)
    parser.add_argument('--permutations_per_batch', type=int, default=5)
    parser.add_argument('--output_type', default='expected_time',
                        choices=['expected_time', 'binary'])
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--num_segments', type=int, default=30)
    parser.add_argument('--nsamples_shap', type=int, default=500)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', type=Path, default=None)
    parser.add_argument('--show_variance_bounds', action='store_true')
    parser.add_argument('--variance_base_samples', type=int, default=50)
    parser.add_argument('--variance_branch_samples', type=int, default=200)
    parser.add_argument('--scenario', default=None,
                        help='Scenario filename stem, e.g. base_chain. Inferred if omitted.')
    args = parser.parse_args()

    sc = None
    if args.scenario is not None:
        sc = next((c for c in ALL_CFGS if c.filename == args.scenario), None)
        if sc is None:
            known = [c.filename for c in ALL_CFGS]
            raise ValueError(f"Unknown scenario '{args.scenario}'. Known: {known}")

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
        variance_base_samples=args.variance_base_samples,
        variance_branch_samples=args.variance_branch_samples,
        scenario=sc,
    )
