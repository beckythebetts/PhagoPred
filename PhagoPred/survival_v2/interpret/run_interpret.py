from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union
import json
import logging
import re

import matplotlib.pyplot as plt
import numpy as np

from PhagoPred.utils.logger import get_logger
from PhagoPred.survival_v2.utils.io import load_model, load_dataset
from PhagoPred.survival_v2.configs.models import RSFCfg
from PhagoPred.survival_v2.configs.losses import BinaryLossCfg
from .SHAP_kernel import KernelSHAP, KernelSHAPResults

log = get_logger()


@dataclass
class RSFImportanceResults:
    """Feature and temporal importance results for RSF models."""
    feature_importance: np.ndarray  # (n_groups,) summed over all windows
    feature_names: list[
        str]  # e.g. ['mean_frame_count', 'std_frame_count', ...]
    temporal_importance_by_scale: dict  # {ws: np.ndarray (num_windows,)}
    window_starts_by_scale: dict  # {ws: np.ndarray (num_windows,)}


def interpret(
    experiment_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    split: Literal["train", "val"] = "val",
    hdf5_paths: Optional[list[Union[str, Path]]] = None,
    num_samples: int = 100,
    num_segments: int = 50,
    nsamples_temporal: int = 300,
    nsamples_feature: int = 200,
    nsamples_temporal_feature: Optional[int] = None,
    output_type: str = "expected_time",
    target_bin: Optional[int] = None,
    mask_value: float = 0.0,
    compute_temporal: bool = True,
    compute_feature: bool = True,
    compute_temporal_feature: bool = True,
    device: str = "cpu",
) -> Optional[KernelSHAPResults]:
    """
    Run perturbation-based SHAP interpretation for a trained model.

    Loads the model and dataset from an experiment directory, runs KernelSHAP
    across a batch of samples, saves importance plots and a results summary
    to output_dir (defaults to experiment_dir/interpret/).

    Args:
        experiment_dir: Path to experiment directory with config.json + model.pkl.
        output_dir: Where to save plots and results. Defaults to
                    experiment_dir/interpret/.
        split: Which dataset split to use ('train' or 'val'). Ignored if
               hdf5_paths is provided.
        hdf5_paths: Override dataset paths (e.g. a held-out test set).
        num_samples: Number of cells to sample from the dataset.
        num_segments: Number of temporal segments for KernelSHAP.
        nsamples_temporal: KernelExplainer perturbation budget for temporal analysis.
        nsamples_feature: KernelExplainer perturbation budget for feature analysis.
        nsamples_temporal_feature: KernelExplainer perturbation budget for joint
            temporal-feature analysis. Defaults to 2 * num_segments * num_features + 2048.
        output_type: Model output to explain ('expected_time', 'cif', 'pmf').
        target_bin: Target bin index when output_type is 'cif' or 'pmf'.
        mask_value: Value substituted when a segment/feature is masked out.
        compute_temporal: Whether to compute temporal importance.
        compute_feature: Whether to compute feature importance.
        compute_temporal_feature: Whether to compute joint temporal-feature importance.
        device: Torch device for deep models.

    Returns:
        KernelSHAPResults with aggregated importance values across all samples.
    """
    log.info(f'Starting kernel SHAP on {experiment_dir}, on device {device}')
    experiment_dir = Path(experiment_dir)
    output_dir = Path(
        output_dir) if output_dir else experiment_dir / "interpret"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {experiment_dir}...")
    model, cfg, _ = load_model(experiment_dir, device=device)

    if isinstance(cfg.model, RSFCfg):
        log.info("Skipping interpretation for RSF model (not supported).")
        return None

    print(f"Loading dataset ({split} split)...")
    dataset = load_dataset(experiment_dir, split=split, hdf5_paths=hdf5_paths)

    explainer = KernelSHAP(
        model=model,
        feature_names=cfg.feature_combo,
        device=device,
    )

    if output_type == "expected_time" and isinstance(cfg.loss, BinaryLossCfg):
        output_type = "binary"
        print("Binary model detected: using output_type='binary'")

    print(f"Running KernelSHAP on {num_samples} samples...")
    results = explainer.analyse_batch(
        dataset=dataset,
        device=device,
        num_samples=num_samples,
        num_segments=num_segments,
        nsamples_temporal=nsamples_temporal,
        nsamples_feature=nsamples_feature,
        nsamples_temporal_feature=nsamples_temporal_feature,
        output_type=output_type,
        target_bin=target_bin,
        mask_value=mask_value,
        compute_temporal=compute_temporal,
        compute_feature=compute_feature,
        compute_temporal_feature=compute_temporal_feature,
    )

    _save_results(results, output_dir)
    _save_plots(explainer, results, output_dir)

    print(f"Results saved to {output_dir}")
    log.info(f'Kenrel SHAP results: {results}')
    return results


def _save_results(results: KernelSHAPResults, output_dir: Path) -> None:
    """Save importance arrays as .npy files and a JSON summary."""
    summary = {}

    if results.temporal_importance is not None:
        np.save(output_dir / "temporal_importance.npy",
                results.temporal_importance)
        np.save(output_dir / "temporal_shap_values.npy",
                results.temporal_shap_values)
        np.save(output_dir / "segment_boundaries.npy",
                results.segment_boundaries)
        summary["temporal_baseline"] = float(
            results.temporal_baseline
        ) if results.temporal_baseline is not None else np.nan

    if results.feature_importance is not None:
        np.save(output_dir / "feature_importance.npy",
                results.feature_importance)
        np.save(output_dir / "feature_shap_values.npy",
                results.feature_shap_values)
        summary["feature_baseline"] = float(
            results.feature_baseline
        ) if results.feature_baseline is not None else np.nan
        summary["feature_names"] = results.feature_names
        summary["feature_importance"] = {
            name: float(imp)
            for name, imp in zip(results.feature_names,
                                 results.feature_importance)
        }

    if results.temporal_feature_importance is not None:
        np.save(
            output_dir / "temporal_feature_importance.npy",
            results.temporal_feature_importance,
        )
        np.save(
            output_dir / "temporal_feature_shap_values.npy",
            results.temporal_feature_shap_values,
        )
        summary["temporal_feature_baseline"] = float(
            results.temporal_feature_baseline
        ) if results.temporal_feature_baseline is not None else np.nan

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def _save_plots(
    explainer: KernelSHAP,
    results: KernelSHAPResults,
    output_dir: Path,
) -> None:
    """Save all available importance plots."""
    if results.temporal_importance is not None:
        fig = explainer.plot_temporal_importance(results)
        fig.savefig(output_dir / "temporal_importance.png",
                    dpi=150,
                    bbox_inches="tight")

    if results.feature_importance is not None:
        fig = explainer.plot_feature_importance(results)
        fig.savefig(output_dir / "feature_importance.png",
                    dpi=150,
                    bbox_inches="tight")

    if results.temporal_feature_importance is not None:
        fig = explainer.plot_temporal_feature_importance(results)
        fig.savefig(output_dir / "temporal_feature_importance.png",
                    dpi=150,
                    bbox_inches="tight")

    if results.temporal_importance is not None and results.feature_importance is not None:
        fig = explainer.plot_summary(results)
        fig.savefig(output_dir / "summary.png", dpi=150, bbox_inches="tight")


# ─── RSF-specific interpretation ────────────────────────────────────────────

_WS_RE = re.compile(r'_ws(\d+)_wi(?:dx)?(\d+)$')


def _interpret_rsf(model, output_dir: Path) -> RSFImportanceResults:
    """Derive feature and temporal importance from RSF feature_importances_."""
    feature_names_all = model.get_feature_names()
    importances_arr = model.model.feature_importances_
    importance_dict = dict(zip(feature_names_all, importances_arr))
    ts = model.temporal_summariser

    base_feat_imp: dict[str, float] = {}
    temporal_imp: dict[int, dict[int, float]] = {
        ws: {}
        for ws in ts.window_sizes
    }

    for name, imp in importance_dict.items():
        m = _WS_RE.search(name)
        if not m:
            continue
        ws = int(m.group(1))
        widx = int(m.group(2))
        base = name[:m.start()]  # e.g. 'mean_frame_count' or 'missingness'

        base_feat_imp[base] = base_feat_imp.get(base, 0.0) + imp
        temporal_imp[ws][widx] = temporal_imp[ws].get(widx, 0.0) + imp

    feature_names = sorted(base_feat_imp)
    feature_importance = np.array([base_feat_imp[n] for n in feature_names])

    temporal_by_scale: dict[int, np.ndarray] = {}
    window_starts_by_scale: dict[int, np.ndarray] = {}
    for ws in ts.window_sizes:
        stride = int(ws * ts.window_overlap)
        num_windows = (ts.full_seq_len - ws) // stride + 1
        temporal_by_scale[ws] = np.array(
            [temporal_imp[ws].get(i, 0.0) for i in range(num_windows)])
        window_starts_by_scale[ws] = np.arange(num_windows) * stride

    results = RSFImportanceResults(
        feature_importance=feature_importance,
        feature_names=feature_names,
        temporal_importance_by_scale=temporal_by_scale,
        window_starts_by_scale=window_starts_by_scale,
    )
    _save_rsf_results(results, output_dir)
    _save_rsf_plots(results, output_dir)
    print(f"RSF importance results saved to {output_dir}")
    return results


def _save_rsf_results(results: RSFImportanceResults, output_dir: Path) -> None:
    np.save(output_dir / "feature_importance.npy", results.feature_importance)
    for ws, imp in results.temporal_importance_by_scale.items():
        np.save(output_dir / f"temporal_importance_ws{ws}.npy", imp)

    summary = {
        "feature_importance":
        dict(zip(results.feature_names, results.feature_importance.tolist())),
        "temporal_importance_by_scale": {
            str(ws): imp.tolist()
            for ws, imp in results.temporal_importance_by_scale.items()
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


def _save_rsf_plots(results: RSFImportanceResults, output_dir: Path) -> None:
    # Feature importance
    n = len(results.feature_names)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.35)))
    idx = np.argsort(results.feature_importance)
    ax.barh(range(n), results.feature_importance[idx], color='steelblue')
    ax.set_yticks(range(n))
    ax.set_yticklabels([results.feature_names[i] for i in idx])
    ax.set_xlabel("Summed importance (across all window scales and positions)")
    ax.set_title("RSF Feature Importance")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "feature_importance.png",
                dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # Temporal importance — one subplot per window scale
    scales = sorted(results.temporal_importance_by_scale)
    fig, axes = plt.subplots(len(scales),
                             1,
                             figsize=(12, 3 * len(scales)),
                             squeeze=False)
    for ax, ws in zip(axes[:, 0], scales):
        starts = results.window_starts_by_scale[ws]
        imp = results.temporal_importance_by_scale[ws]
        stride = starts[1] - starts[0] if len(starts) > 1 else ws
        ax.bar(starts,
               imp,
               width=stride * 0.9,
               align='edge',
               color='coral',
               edgecolor='darkred',
               alpha=0.7)
        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Summed importance")
        ax.set_title(f"Window scale ws={ws}")
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle("RSF Temporal Importance by Window Scale", fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "temporal_importance.png",
                dpi=150,
                bbox_inches="tight")
    plt.close(fig)
