"""
Perturbation-based SHAP for survival models using KernelExplainer.

This module provides SHAP analysis that works with any pooling mechanism,
including LastPooling where gradient-based methods show importance only at
the final timestep. It computes temporal and feature importance separately
by reducing the input to 1D masks.

Key advantages over gradient-based SHAP:
- Works with any architecture (no gradient flow requirements)
- Directly measures prediction change when inputs are masked
- Can handle temporal and feature importance independently
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm

try:
    import shap
except ImportError:
    shap = None

from PhagoPred.survival_v2.models.base import SurvivalModel
from PhagoPred.survival_v2.data.dataset import CellDataset, collate_fn


@dataclass
class KernelSHAPResults:
    """Container for kernel SHAP analysis results."""
    # Temporal analysis
    temporal_shap_values: Optional[np.ndarray] = None  # (num_samples, num_segments)
    temporal_importance: Optional[np.ndarray] = None   # (num_segments,)
    segment_boundaries: Optional[np.ndarray] = None    # (num_segments + 1,)

    # Feature analysis
    feature_shap_values: Optional[np.ndarray] = None   # (num_samples, num_features)
    feature_importance: Optional[np.ndarray] = None    # (num_features,)
    feature_names: Optional[list[str]] = None

    # Baseline values
    temporal_baseline: Optional[float] = None
    feature_baseline: Optional[float] = None


class TemporalMaskWrapper:
    """
    Wrapper for computing importance of temporal segments.

    Takes a 1D mask of length num_segments and applies it to the input
    by masking all features at the corresponding timesteps.
    """

    def __init__(
        self,
        model: SurvivalModel,
        base_input: torch.Tensor,
        lengths: torch.Tensor,
        num_segments: int,
        output_type: Literal["expected_time", "cif", "pmf"] = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        device: str = "cpu",
    ):
        """
        Args:
            model: Survival model to explain
            base_input: Input tensor (1, T, F)
            lengths: Sequence lengths (1,)
            num_segments: Number of temporal segments to divide the sequence into
            output_type: Which model output to use
            target_bin: For pmf/cif, which bin to explain
            mask_value: Value to use when masking (0.0 or mean)
            device: Computation device
        """
        self.model = model
        self.model.eval()
        self.base_input = base_input.to(device)
        self.lengths = lengths.to(device)
        self.device = device
        self.output_type = output_type
        self.target_bin = target_bin
        self.mask_value = mask_value

        self.T = base_input.shape[1]
        self.num_segments = num_segments
        self.segment_size = self.T // num_segments

        # Precompute segment boundaries
        self.segment_boundaries = np.linspace(0, self.T, num_segments + 1, dtype=int)

    def __call__(self, segment_masks: np.ndarray) -> np.ndarray:
        """
        Apply segment masks and return model predictions.

        Args:
            segment_masks: (N, num_segments) array of binary/continuous masks

        Returns:
            predictions: (N,) array of model outputs
        """
        outputs = []

        for mask in segment_masks:
            # Expand segment mask to full timeline
            full_mask = np.zeros(self.T)
            for i, m in enumerate(mask):
                start = self.segment_boundaries[i]
                end = self.segment_boundaries[i + 1]
                full_mask[start:end] = m

            # Apply mask: where mask=0, use mask_value; where mask=1, keep original
            mask_tensor = torch.tensor(full_mask, device=self.device, dtype=torch.float32)
            masked_input = self.base_input.clone()

            # Blend between mask_value and original based on mask
            masked_input = masked_input * mask_tensor.view(1, -1, 1) + \
                          self.mask_value * (1 - mask_tensor.view(1, -1, 1))

            with torch.no_grad():
                pred = self.model(masked_input, self.lengths, return_attention=False)
                if isinstance(pred, tuple):
                    pred = pred[0]

                pred = self._extract_output(pred)

            outputs.append(pred.cpu().numpy().flatten())

        # Return shape (N, 1) - KernelExplainer requires 2D output
        return np.array(outputs).reshape(-1, 1)

    def _extract_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract the appropriate scalar output from model logits."""
        if self.output_type == "expected_time":
            return self.model.predict_expected_time(logits)
        elif self.output_type == "cif":
            cif = self.model.predict_cif(logits)
            if self.target_bin is not None:
                return cif[:, self.target_bin]
            return cif[:, -1]  # Final CIF value
        elif self.output_type == "pmf":
            pmf = self.model.predict_pmf(logits)
            if self.target_bin is not None:
                return pmf[:, self.target_bin]
            return pmf.max(dim=1).values  # Max probability
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class FeatureMaskWrapper:
    """
    Wrapper for computing importance of each feature.

    Takes a 1D mask of length num_features and applies it across all timesteps.
    """

    def __init__(
        self,
        model: SurvivalModel,
        base_input: torch.Tensor,
        lengths: torch.Tensor,
        output_type: Literal["expected_time", "cif", "pmf"] = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        device: str = "cpu",
    ):
        """
        Args:
            model: Survival model to explain
            base_input: Input tensor (1, T, F)
            lengths: Sequence lengths (1,)
            output_type: Which model output to use
            target_bin: For pmf/cif, which bin to explain
            mask_value: Value to use when masking
            device: Computation device
        """
        self.model = model
        self.model.eval()
        self.base_input = base_input.to(device)
        self.lengths = lengths.to(device)
        self.device = device
        self.output_type = output_type
        self.target_bin = target_bin
        self.mask_value = mask_value

        self.F = base_input.shape[2]

    def __call__(self, feature_masks: np.ndarray) -> np.ndarray:
        """
        Apply feature masks and return model predictions.

        Args:
            feature_masks: (N, num_features) array of binary/continuous masks

        Returns:
            predictions: (N,) array of model outputs
        """
        outputs = []

        for mask in feature_masks:
            # Apply mask: (F,) -> (1, 1, F) broadcast over timesteps
            mask_tensor = torch.tensor(mask, device=self.device, dtype=torch.float32)
            masked_input = self.base_input.clone()

            # Blend between mask_value and original based on mask
            masked_input = masked_input * mask_tensor.view(1, 1, -1) + \
                          self.mask_value * (1 - mask_tensor.view(1, 1, -1))

            with torch.no_grad():
                pred = self.model(masked_input, self.lengths, return_attention=False)
                if isinstance(pred, tuple):
                    pred = pred[0]

                pred = self._extract_output(pred)

            outputs.append(pred.cpu().numpy().flatten())

        # Return shape (N, 1) - KernelExplainer requires 2D output
        return np.array(outputs).reshape(-1, 1)

    def _extract_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract the appropriate scalar output from model logits."""
        if self.output_type == "expected_time":
            return self.model.predict_expected_time(logits)
        elif self.output_type == "cif":
            cif = self.model.predict_cif(logits)
            if self.target_bin is not None:
                return cif[:, self.target_bin]
            return cif[:, -1]
        elif self.output_type == "pmf":
            pmf = self.model.predict_pmf(logits)
            if self.target_bin is not None:
                return pmf[:, self.target_bin]
            return pmf.max(dim=1).values
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class KernelSHAP:
    """
    Perturbation-based SHAP analysis for survival models.

    Uses KernelExplainer to compute SHAP values by masking temporal segments
    or features. Works with any model architecture including LastPooling.

    Example:
        >>> explainer = KernelSHAP(model, feature_names=['f0', 'f1', 'f2'])
        >>> results = explainer.analyse(x, lengths, num_segments=50, nsamples=500)
        >>> explainer.plot_summary(results)
    """

    def __init__(
        self,
        model: SurvivalModel,
        feature_names: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize KernelSHAP explainer.

        Args:
            model: Trained SurvivalModel
            feature_names: Names of features for visualization
            device: Computation device
        """
        if shap is None:
            raise ImportError("shap package not installed. Install with: pip install shap")

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.feature_names = feature_names

    def compute_temporal_importance(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        num_segments: int = 50,
        nsamples: int = 500,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Compute SHAP values for temporal segments.

        Args:
            x: Input tensor (1, T, F) - single sample
            lengths: Sequence length (1,)
            num_segments: Number of temporal segments to analyze
            nsamples: Number of samples for KernelExplainer
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking segments
            show_progress: Whether to show progress bar

        Returns:
            (shap_values, segment_boundaries, baseline_value)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor([lengths])
        elif lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        # Create wrapper
        wrapper = TemporalMaskWrapper(
            model=self.model,
            base_input=x,
            lengths=lengths,
            num_segments=num_segments,
            output_type=output_type,
            target_bin=target_bin,
            mask_value=mask_value,
            device=self.device,
        )

        # Background: all segments MASKED (zeros) - represents "no information"
        background = np.zeros((1, num_segments))

        # Create explainer
        explainer = shap.KernelExplainer(wrapper, background)

        # Explain the full input (all segments = 1, i.e., all present)
        test_input = np.ones((1, num_segments))
        shap_values = explainer.shap_values(test_input, nsamples=nsamples, silent=not show_progress)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        baseline = explainer.expected_value
        if isinstance(baseline, np.ndarray):
            baseline = baseline[0]

        return shap_values.squeeze(), wrapper.segment_boundaries, baseline

    def compute_feature_importance(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        nsamples: int = 500,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, float]:
        """
        Compute SHAP values for each feature.

        Args:
            x: Input tensor (1, T, F) - single sample
            lengths: Sequence length (1,)
            nsamples: Number of samples for KernelExplainer
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking features
            show_progress: Whether to show progress bar

        Returns:
            (shap_values, baseline_value)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor([lengths])
        elif lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        num_features = x.shape[2]

        # Create wrapper
        wrapper = FeatureMaskWrapper(
            model=self.model,
            base_input=x,
            lengths=lengths,
            output_type=output_type,
            target_bin=target_bin,
            mask_value=mask_value,
            device=self.device,
        )

        # Background: all features MASKED (zeros) - represents "no information"
        background = np.zeros((1, num_features))

        # Create explainer
        explainer = shap.KernelExplainer(wrapper, background)

        # Explain the full input (all features = 1, i.e., all present)
        test_input = np.ones((1, num_features))
        shap_values = explainer.shap_values(test_input, nsamples=nsamples, silent=not show_progress)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        baseline = explainer.expected_value
        if isinstance(baseline, np.ndarray):
            baseline = baseline[0]

        return shap_values.squeeze(), baseline

    def analyse(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        num_segments: int = 50,
        nsamples_temporal: int = 500,
        nsamples_feature: int = 200,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        compute_temporal: bool = True,
        compute_feature: bool = True,
        show_progress: bool = True,
    ) -> KernelSHAPResults:
        """
        Full SHAP analysis for a single sample.

        Args:
            x: Input tensor (1, T, F) or (T, F)
            lengths: Sequence length
            num_segments: Number of temporal segments
            nsamples_temporal: KernelExplainer samples for temporal analysis
            nsamples_feature: KernelExplainer samples for feature analysis
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking
            compute_temporal: Whether to compute temporal importance
            compute_feature: Whether to compute feature importance
            show_progress: Whether to show progress

        Returns:
            KernelSHAPResults with all computed values
        """
        results = KernelSHAPResults(feature_names=self.feature_names)

        if compute_temporal:
            if show_progress:
                print("Computing temporal importance...")
            temporal_shap, boundaries, temporal_baseline = self.compute_temporal_importance(
                x, lengths, num_segments, nsamples_temporal,
                output_type, target_bin, mask_value, show_progress
            )
            results.temporal_shap_values = temporal_shap.reshape(1, -1)
            results.temporal_importance = np.abs(temporal_shap)
            results.segment_boundaries = boundaries
            results.temporal_baseline = temporal_baseline

        if compute_feature:
            if show_progress:
                print("Computing feature importance...")
            feature_shap, feature_baseline = self.compute_feature_importance(
                x, lengths, nsamples_feature,
                output_type, target_bin, mask_value, show_progress
            )
            results.feature_shap_values = feature_shap.reshape(1, -1)
            results.feature_importance = np.abs(feature_shap)
            results.feature_baseline = feature_baseline

        return results

    def analyse_batch(
        self,
        dataset: Union[CellDataset, torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
        num_samples: int = 10,
        num_segments: int = 50,
        nsamples_temporal: int = 300,
        nsamples_feature: int = 100,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        max_len: int = 1000,
    ) -> KernelSHAPResults:
        """
        Analyse multiple samples and aggregate results.

        Args:
            dataset: CellDataset or tensor of shape (N, T, F)
            lengths: Sequence lengths if dataset is a tensor
            num_samples: Number of samples to analyse
            num_segments: Number of temporal segments
            nsamples_temporal: KernelExplainer samples per sample (temporal)
            nsamples_feature: KernelExplainer samples per sample (feature)
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking
            max_len: Maximum sequence length

        Returns:
            KernelSHAPResults with aggregated importance values
        """
        if isinstance(dataset, CellDataset):
            # Sample from dataset
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=num_samples,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, dataset, max_seq_len=max_len),
            )
            batch = next(iter(dataloader))
            x_batch = batch['features']
            lengths_batch = batch['length']
        else:
            x_batch = dataset[:num_samples]
            lengths_batch = lengths[:num_samples] if lengths is not None else \
                           torch.full((num_samples,), x_batch.shape[1])

        all_temporal = []
        all_feature = []
        segment_boundaries = None

        for i in tqdm(range(min(num_samples, len(x_batch))), desc="Analysing samples"):
            x_i = x_batch[i:i+1]
            l_i = lengths_batch[i:i+1]

            result = self.analyse(
                x_i, l_i,
                num_segments=num_segments,
                nsamples_temporal=nsamples_temporal,
                nsamples_feature=nsamples_feature,
                output_type=output_type,
                target_bin=target_bin,
                mask_value=mask_value,
                show_progress=False,
            )

            all_temporal.append(result.temporal_shap_values)
            all_feature.append(result.feature_shap_values)
            segment_boundaries = result.segment_boundaries

        # Aggregate
        temporal_shap = np.vstack(all_temporal)
        feature_shap = np.vstack(all_feature)

        return KernelSHAPResults(
            temporal_shap_values=temporal_shap,
            temporal_importance=np.abs(temporal_shap).mean(axis=0),
            segment_boundaries=segment_boundaries,
            feature_shap_values=feature_shap,
            feature_importance=np.abs(feature_shap).mean(axis=0),
            feature_names=self.feature_names,
        )

    # ==================== Visualization Methods ====================

    def plot_temporal_importance(
        self,
        results: KernelSHAPResults,
        title: str = "Temporal Importance (Kernel SHAP)",
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot temporal importance across segments."""
        fig, ax = plt.subplots(figsize=figsize)

        importance = results.temporal_importance
        boundaries = results.segment_boundaries

        # Use segment centers for x-axis
        centers = (boundaries[:-1] + boundaries[1:]) / 2
        widths = np.diff(boundaries)

        ax.bar(centers, importance, width=widths * 0.9,
               color='steelblue', edgecolor='navy', alpha=0.7)

        ax.set_xlabel("Time (frames)")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_feature_importance(
        self,
        results: KernelSHAPResults,
        title: str = "Feature Importance (Kernel SHAP)",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=figsize)

        importance = results.feature_importance
        names = results.feature_names or [f"Feature_{i}" for i in range(len(importance))]

        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_names = [names[i] for i in sorted_idx]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance)))
        ax.barh(range(len(importance)), sorted_importance[::-1], color=colors)
        ax.set_yticks(range(len(importance)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_summary(
        self,
        results: KernelSHAPResults,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot combined temporal and feature importance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Temporal importance
        if results.temporal_importance is not None:
            importance = results.temporal_importance
            boundaries = results.segment_boundaries
            centers = (boundaries[:-1] + boundaries[1:]) / 2
            widths = np.diff(boundaries)

            ax1.bar(centers, importance, width=widths * 0.9,
                   color='coral', edgecolor='darkred', alpha=0.7)
            ax1.set_xlabel("Time (frames)")
            ax1.set_ylabel("Mean |SHAP value|")
            ax1.set_title("Temporal Importance")
            ax1.grid(axis='y', alpha=0.3)

        # Feature importance
        if results.feature_importance is not None:
            importance = results.feature_importance
            names = results.feature_names or [f"F{i}" for i in range(len(importance))]

            sorted_idx = np.argsort(importance)[::-1]
            sorted_importance = importance[sorted_idx]
            sorted_names = [names[i] for i in sorted_idx]

            ax2.barh(range(len(importance)), sorted_importance[::-1], color='steelblue')
            ax2.set_yticks(range(len(importance)))
            ax2.set_yticklabels(sorted_names[::-1])
            ax2.set_xlabel("Mean |SHAP value|")
            ax2.set_title("Feature Importance")
            ax2.grid(axis='x', alpha=0.3)

        plt.suptitle("Kernel SHAP Analysis (Perturbation-Based)", fontsize=12, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_shap_values(
        self,
        results: KernelSHAPResults,
        sample_idx: int = 0,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot raw SHAP values (positive and negative) for a sample."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Temporal SHAP values
        if results.temporal_shap_values is not None:
            shap_vals = results.temporal_shap_values[sample_idx]
            boundaries = results.segment_boundaries
            centers = (boundaries[:-1] + boundaries[1:]) / 2

            colors = ['coral' if v > 0 else 'steelblue' for v in shap_vals]
            ax1.bar(centers, shap_vals, width=np.diff(boundaries) * 0.9,
                   color=colors, edgecolor='black', alpha=0.7)
            ax1.axhline(0, color='black', linewidth=0.5)
            ax1.set_xlabel("Time (frames)")
            ax1.set_ylabel("SHAP value")
            ax1.set_title("Temporal SHAP Values")
            ax1.grid(axis='y', alpha=0.3)

        # Feature SHAP values
        if results.feature_shap_values is not None:
            shap_vals = results.feature_shap_values[sample_idx]
            names = results.feature_names or [f"F{i}" for i in range(len(shap_vals))]

            colors = ['coral' if v > 0 else 'steelblue' for v in shap_vals]
            ax2.barh(range(len(shap_vals)), shap_vals, color=colors, alpha=0.7)
            ax2.axvline(0, color='black', linewidth=0.5)
            ax2.set_yticks(range(len(shap_vals)))
            ax2.set_yticklabels(names)
            ax2.set_xlabel("SHAP value")
            ax2.set_title("Feature SHAP Values")
            ax2.grid(axis='x', alpha=0.3)

        plt.suptitle(f"SHAP Values (Sample {sample_idx})", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


# ==================== Convenience Functions ====================

def quick_temporal_importance(
    model: SurvivalModel,
    x: torch.Tensor,
    lengths: torch.Tensor,
    num_segments: int = 50,
    nsamples: int = 500,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quick temporal importance analysis for a single sample.

    Returns:
        (importance, segment_boundaries)
    """
    explainer = KernelSHAP(model, device=device)
    shap_vals, boundaries, _ = explainer.compute_temporal_importance(
        x, lengths, num_segments, nsamples
    )
    return np.abs(shap_vals), boundaries


def quick_feature_importance(
    model: SurvivalModel,
    x: torch.Tensor,
    lengths: torch.Tensor,
    feature_names: Optional[list[str]] = None,
    nsamples: int = 200,
    device: str = "cpu",
) -> np.ndarray:
    """
    Quick feature importance analysis for a single sample.

    Returns:
        importance array of shape (num_features,)
    """
    explainer = KernelSHAP(model, feature_names=feature_names, device=device)
    shap_vals, _ = explainer.compute_feature_importance(x, lengths, nsamples)
    return np.abs(shap_vals)
