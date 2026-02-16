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
from abc import abstractmethod
from dataclasses import dataclass, asdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Literal, Optional, Union
from tqdm import tqdm
import shap

from PhagoPred.survival_v2.models.base import SurvivalModel
from PhagoPred.survival_v2.data.dataset import CellDataset, collate_fn


@dataclass
class KernelSHAPResults:
    """Container for kernel SHAP analysis results."""
    # Temporal analysis
    temporal_shap_values: Optional[np.ndarray] = None  # (num_samples, num_segments)
    temporal_importance: Optional[np.ndarray] = None   # (num_segments,)
    segment_boundaries: Optional[np.ndarray] = None    # (num_segments + 1,)
    temporal_baseline: Optional[float] = None

    # Feature analysis
    feature_shap_values: Optional[np.ndarray] = None   # (num_samples, num_features)
    feature_importance: Optional[np.ndarray] = None    # (num_features,)
    feature_names: Optional[list[str]] = None
    feature_baseline: Optional[float] = None

    # Temporal-feature analysis (if implemented)
    temporal_feature_shap_values: Optional[np.ndarray] = None  # (num_samples, num_segments, num_features)
    temporal_feature_importance: Optional[np.ndarray] = None   # (num_segments, num_features)
    temporal_feature_baseline: Optional[float] = None
    
    def asdict(self):
        return asdict(self)
    
class MaskWrapperBase:
    """Base class for mask wrappers used in Kernel SHAP analysis.
    Allows modesl to be called with different masks applied.
    
    Attributes:
        model: The survival model to explain
        base_input: The original input tensor (1, T, F)
        lengths: Sequence length tensor (1,)
        output_type: Which model output to explain ("expected_time", "cif", "pmf")
        target_bin: For "cif" or "pmf", which bin to explain (if None, uses final bin or max)
        mask_value: Value to use when masking (e.g., 0.0 or mean value)
        device: Computation device
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
        self.model = model
        self.model.eval()
        self.base_input = base_input.to(device)
        self.lengths = lengths.to(device)
        self.device = device
        self.output_type = output_type
        self.target_bin = target_bin
        self.mask_value = mask_value
        
    def __call__(self, masks: np.ndarray) -> np.ndarray:
        """Apply masks and return model predictions."""
        outputs = []

        for mask in masks:
            masked_input = self._apply_mask(mask)
            with torch.no_grad():
                pred = self.model(masked_input, self.lengths, return_attention=False)
                if isinstance(pred, tuple):
                    pred = pred[0]

                pred = self._extract_output(pred)

            outputs.append(pred.cpu().numpy().flatten())

        # Return shape (N, 1) - KernelExplainer requires 2D output
        return np.array(outputs).reshape(-1, 1)
    
    @abstractmethod
    def _apply_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Apply the given mask to the base input and return the masked input."""
        pass
    
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
        
class TemporalMaskWrapper(MaskWrapperBase):
    """
    Wrapper for computing importance of temporal segments.

    Takes a 1D mask of length num_segments and applies it to the input
    by masking all features at the corresponding timesteps.
    """

    def __init__(self, *args, num_segments: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.T = self.base_input.shape[1]
        # self.T = int(self.lengths[0].item())

        self.segment_boundaries = np.linspace(0, self.T, num_segments + 1, dtype=int)

    def _apply_mask(self, mask: np.ndarray) -> torch.Tensor:
        full_mask = np.zeros(self.T)
        for i, m in enumerate(mask):
            start = self.segment_boundaries[i]
            end = self.segment_boundaries[i + 1]
            full_mask[start:end] = m

        mask_tensor = torch.tensor(full_mask, device=self.device, dtype=torch.float32)
        masked_input = self.base_input.clone()

        masked_input = masked_input * mask_tensor.view(1, -1, 1) + \
                        self.mask_value * (1 - mask_tensor.view(1, -1, 1))
        return masked_input

class FeatureMaskWrapper(MaskWrapperBase):
    """
    Wrapper for computing importance of each feature.

    Takes a 1D mask of length num_features and applies it across all timesteps.
    """

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        self.F = self.base_input.shape[2]

    def _apply_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Apply a feature mask across all timesteps.
        Args:
            mask: np.ndarray of shape (num_features,) with values in {0, 1}
        """
        mask_tensor = torch.tensor(mask, device=self.device, dtype=torch.float32)
        masked_input = self.base_input.clone()

        # Blend between mask_value and original based on mask
        masked_input = masked_input * mask_tensor.view(1, 1, -1) + \
                        self.mask_value * (1 - mask_tensor.view(1, 1, -1))
        return masked_input

class TemporalFeatureMaskWrapper(MaskWrapperBase):
    def __init__(self, *args, num_segments: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.T = self.base_input.shape[1]
        # self.T = int(self.lengths[0].item())
        self.F = self.base_input.shape[2]

        self.segment_boundaries = np.linspace(0, self.T, num_segments + 1, dtype=int)

    def _apply_mask(self, mask: np.ndarray) -> torch.Tensor:
        """Apply a 2D mask of shape (num_segments, num_features) to the input.
        Args
        ----
            mask: np.ndarray of shape (num_segments, num_features) with values in {0, 1}
        """
        mask = mask.reshape(self.num_segments, self.F)  # Ensure correct shape
        full_mask = np.zeros((self.T, self.F))
        for i, m in enumerate(mask):
            start = self.segment_boundaries[i]
            end = self.segment_boundaries[i + 1]
            full_mask[start:end, :] = m 
        mask_tensor = torch.tensor(full_mask, device=self.device, dtype=torch.float32)
        masked_input = self.base_input.clone()  
        masked_input = masked_input * mask_tensor + self.mask_value * (1 - mask_tensor)
        return masked_input

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

    def compute_importance(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        num_segments: Optional[int] = 50,
        nsamples: int = 500,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        show_progress: bool = True,
        importance_type: Literal["temporal", "feature", "temporal_feature"] = "temporal",
        ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute SHAP values for either temporal segments, features, or both.
        Args:
            x: Input tensor (1, T, F) - single sample
            lengths: Sequence length (1,)
            num_segments: Number of temporal segments to analyze
            nsamples: Number of samples for KernelExplainer
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking
            show_progress: Whether to show progress bar
            importance_type: Which importance to compute ("temporal", "feature", "temporal_feature")
        Returns:
            (shap_values, baseline_value, segment_boundaries) for temporal importance
            (shap_values, baseline_value) for feature importance
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor([lengths])
        elif lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)
    
        kwargs = dict(
            model=self.model,
            base_input=x,
            lengths=lengths,
            output_type=output_type,
            target_bin=target_bin,
            mask_value=mask_value,
            device=self.device,
        )

        if importance_type == "feature":
            background = np.zeros((1, x.shape[2]))
            wrapper = FeatureMaskWrapper(**kwargs)
        elif importance_type == "temporal":
            background = np.zeros((1, num_segments))
            wrapper = TemporalMaskWrapper(**kwargs, num_segments=num_segments)
        elif importance_type == "temporal_feature":
            background = np.zeros((1, num_segments*x.shape[2]))
            wrapper = TemporalFeatureMaskWrapper(**kwargs, num_segments=num_segments)
            
        explainer = shap.KernelExplainer(wrapper, background)

        test_input = np.ones_like(background)

        # l1_reg = False if importance_type == "temporal_feature" else "num_features(10)"
        shap_values = explainer.shap_values(test_input, nsamples="auto", silent=not show_progress)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        baseline = explainer.expected_value
        if isinstance(baseline, np.ndarray):
            baseline = baseline[0]

        return shap_values.squeeze(), baseline, wrapper.segment_boundaries if importance_type in ["temporal", "temporal_feature"] else None

    def analyse(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        num_segments: int = 50,
        nsamples_temporal: int = 500,
        nsamples_feature: int = 200,
        nsamples_temporal_feature: Optional[int] = None,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        compute_temporal: bool = True,
        compute_feature: bool = True,
        compute_temporal_feature: bool = True,
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
            nsamples_temporal_feature: KernelExplainer samples for temporal-feature
                analysis. If None, defaults to 2 * num_segments * num_features + 2048.
            output_type: Model output to explain
            target_bin: Target bin for pmf/cif
            mask_value: Value to use when masking
            compute_temporal: Whether to compute temporal importance
            compute_feature: Whether to compute feature importance
            compute_temporal_feature: Whether to compute temporal-feature importance
            show_progress: Whether to show progress

        Returns:
            KernelSHAPResults with all computed values
        """
        results = KernelSHAPResults(feature_names=self.feature_names)

        if compute_temporal:
            if show_progress:
                print("Computing temporal importance...")
            temporal_shap, temporal_baseline, boundaries = self.compute_importance(
                x, lengths, num_segments, nsamples_temporal,
                output_type, target_bin, mask_value, show_progress, 'temporal'
            )
            results.temporal_shap_values = temporal_shap.reshape(1, -1)
            results.temporal_importance = np.abs(temporal_shap)
            results.segment_boundaries = boundaries
            results.temporal_baseline = temporal_baseline

        if compute_feature:
            if show_progress:
                print("Computing feature importance...")
            feature_shap, feature_baseline, _ = self.compute_importance(
                x, lengths, num_segments, nsamples_feature,
                output_type, target_bin, mask_value, show_progress, 'feature'
            )
            results.feature_shap_values = feature_shap.reshape(1, -1)
            results.feature_importance = np.abs(feature_shap)
            results.feature_baseline = feature_baseline

        if compute_temporal_feature:
            if show_progress:
                print("Computing temporal-feature importance...")
            if x.dim() == 2:
                num_features = x.shape[1]
            else:
                num_features = x.shape[2]
            if nsamples_temporal_feature is None:
                nsamples_temporal_feature = 2 * num_segments * num_features + 2048
            temporal_feature_shap, temporal_feature_baseline, boundaries = self.compute_importance(
                x, lengths, num_segments, nsamples_temporal_feature,
                output_type, target_bin, mask_value, show_progress, 'temporal_feature'
            )
            temporal_feature_shap_2d = temporal_feature_shap.reshape(num_segments, -1)
            results.temporal_feature_shap_values = temporal_feature_shap_2d.reshape(1, num_segments, -1)
            results.temporal_feature_importance = np.abs(temporal_feature_shap_2d)
            results.temporal_feature_baseline = temporal_feature_baseline
        
        return results

    def analyse_batch(
        self,
        dataset: Union[CellDataset, torch.Tensor],
        lengths: Optional[torch.Tensor] = None,
        num_samples: int = 10,
        num_segments: int = 50,
        nsamples_temporal: int = 300,
        nsamples_feature: int = 100,
        nsamples_temporal_feature: Optional[int] = None,
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
            nsamples_temporal_feature: KernelExplainer samples per sample
                (temporal-feature). If None, defaults to 2 * num_segments * num_features + 2048.
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
        all_temporal_feature = []
        segment_boundaries = None

        for i in tqdm(range(min(num_samples, len(x_batch))), desc="Analysing samples"):
            x_i = x_batch[i:i+1]
            l_i = lengths_batch[i:i+1]

            result = self.analyse(
                x_i, l_i,
                num_segments=num_segments,
                nsamples_temporal=nsamples_temporal,
                nsamples_feature=nsamples_feature,
                nsamples_temporal_feature=nsamples_temporal_feature,
                output_type=output_type,
                target_bin=target_bin,
                mask_value=mask_value,
                show_progress=False,
            )

            all_temporal.append(result.temporal_shap_values)
            all_feature.append(result.feature_shap_values)
            if result.temporal_feature_shap_values is not None:
                all_temporal_feature.append(result.temporal_feature_shap_values)
            segment_boundaries = result.segment_boundaries

        # Aggregate
        temporal_shap = np.vstack(all_temporal)
        feature_shap = np.vstack(all_feature)

        tf_shap = None
        tf_importance = None
        if all_temporal_feature:
            tf_shap = np.vstack(all_temporal_feature)
            tf_importance = np.abs(tf_shap).mean(axis=0)

        return KernelSHAPResults(
            temporal_shap_values=temporal_shap,
            temporal_importance=np.abs(temporal_shap).mean(axis=0),
            segment_boundaries=segment_boundaries,
            feature_shap_values=feature_shap,
            feature_importance=np.abs(feature_shap).mean(axis=0),
            feature_names=self.feature_names,
            temporal_feature_shap_values=tf_shap,
            temporal_feature_importance=tf_importance,
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

    def plot_temporal_feature_importance(
        self,
        results: KernelSHAPResults,
        title: str = "Temporal-Feature Importance (Kernel SHAP)",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot temporal-feature importance as a heatmap."""
        fig, ax = plt.subplots(figsize=figsize)

        importance = results.temporal_feature_importance
        feature_names = results.feature_names or [f"Feature_{i}" for i in range(importance.shape[1])]
        segment_labels = [f"Seg {i}" for i in range(importance.shape[0])]

        im = ax.imshow(importance.T, aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names)
        ax.set_xticks(range(len(segment_labels)))
        ax.set_xticklabels(segment_labels, rotation=45)
        ax.set_xlabel("Temporal Segments")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Mean |SHAP value|")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig
    
    def plot_summary(
        self,
        results: KernelSHAPResults,
        figsize: tuple = (14, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot combined temporal and feature importance."""
        if results.temporal_feature_importance is None:
            fig, (ax_temp, ax_feat) = plt.subplots(1, 3, figsize=figsize)
        else:
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 2], hspace=0.3)
            ax_temp = fig.add_subplot(gs[0, 0])
            ax_feat = fig.add_subplot(gs[0, 1])
            ax_both = fig.add_subplot(gs[1, :])

        # Temporal importance
        if results.temporal_importance is not None:
            importance = results.temporal_importance
            boundaries = results.segment_boundaries
            centers = (boundaries[:-1] + boundaries[1:]) / 2
            widths = np.diff(boundaries)

            ax_temp.bar(centers, importance, width=widths * 0.9,
                   color='coral', edgecolor='darkred', alpha=0.7)
            ax_temp.set_xlabel("Time (frames)")
            ax_temp.set_ylabel("Mean |SHAP value|")
            ax_temp.set_title("Temporal Importance")
            ax_temp.grid(axis='y', alpha=0.3)

        # Feature importance
        if results.feature_importance is not None:
            importance = results.feature_importance
            names = results.feature_names or [f"F{i}" for i in range(len(importance))]

            sorted_idx = np.argsort(importance)[::-1]
            sorted_importance = importance[sorted_idx]
            sorted_names = [names[i] for i in sorted_idx]

            ax_feat.barh(range(len(importance)), sorted_importance[::-1], color='steelblue')
            ax_feat.set_yticks(range(len(importance)))
            ax_feat.set_yticklabels(sorted_names[::-1])
            ax_feat.set_xlabel("Mean |SHAP value|")
            ax_feat.set_title("Feature Importance")
            ax_feat.grid(axis='x', alpha=0.3)
        
        if results.temporal_feature_importance is not None:
            importance = results.temporal_feature_importance
            feature_names = results.feature_names or [f"Feature_{i}" for i in range(importance.shape[1])]
            segment_labels = [f"Seg {i}" for i in range(importance.shape[0])]

            im = ax_both.imshow(importance.T, aspect='auto', cmap='viridis')
            ax_both.set_yticks(range(len(feature_names)))
            ax_both.set_yticklabels(feature_names)
            ax_both.set_xticks(range(len(segment_labels)))
            ax_both.set_xticklabels(segment_labels, rotation=45)
            ax_both.set_xlabel("Temporal Segments")
            ax_both.set_title("Temporal-Feature Importance")
            fig.colorbar(im, ax=ax_both, label="Mean |SHAP value|")

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

    def plot_sample_explanation(
        self,
        sample: dict,
        bin_edges,
        num_segments: int = 20,
        nsamples: int = None,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        mask_value: float = 0.0,
        feature_names: Optional[list[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compute SHAP temporal-feature importance for a single sample and plot
        it alongside observed features and the predicted (+ true) PMF.

        Layout mirrors visualise_prediction but replaces the attention panel
        with a temporal-feature SHAP heatmap.

        Args:
            sample: dict with keys (same as visualise_prediction):
                - features: (seq_len, num_features)
                - time_to_event: scalar
                - event_indicator: scalar (1=event, 0=censored)
                - (optional) binned_pmf: true PMF
                - (optional) length: sequence length
                - (optional) landmark_frame: landmark frame offset
            bin_edges: bin edge positions for the PMF
            num_segments: number of temporal segments for SHAP analysis
            nsamples: KernelExplainer perturbation budget
                (default: 2 * num_segments * num_features + 2048)
            output_type: model output to explain ("expected_time", "cif", "pmf")
            target_bin: target bin for pmf/cif output_type
            mask_value: value used when masking features
            feature_names: list of feature names (falls back to self.feature_names)
            start_frame: x-axis offset for feature time series
            save_path: path to save figure, or None
        """
        # Helper function to convert to numpy if needed
        def to_numpy(x):
            if x is None:
                return None
            if hasattr(x, 'cpu'):  # torch tensor
                return x.cpu().numpy()
            elif isinstance(x, np.ndarray):
                return x
            else:
                return np.array(x)

        # Convert all inputs to numpy arrays
        features = to_numpy(sample['features'])
        true_pmf = to_numpy(sample.get('binned_pmf'))
        time_to_event = to_numpy(sample['time_to_event'])
        length = to_numpy(sample.get('length', None))
        landmark_frame = to_numpy(sample.get('landmark_frame', 0))
        start_frame = to_numpy(sample.get('start_frame', 0))
        if length is None or (isinstance(length, np.ndarray) and length.size == 0):
            length = len(features)
        elif isinstance(length, np.ndarray):
            length = int(length.item())
        else:
            length = int(length)
        event_indicator = to_numpy(sample['event_indicator'])
        if isinstance(event_indicator, np.ndarray):
            event_indicator = event_indicator.item()

        names = feature_names or self.feature_names or [
            f"F{i}" for i in range(features.shape[1])
        ]

        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        lengths = torch.tensor([length])

        with torch.no_grad():
            logits = self.model(x.to(self.device), lengths.to(self.device))
            if isinstance(logits, tuple):
                logits = logits[0]
            predicted_pmf = self.model.predict_pmf(logits).cpu().numpy().squeeze()

        num_features = x.shape[2]
        if nsamples is None:
            nsamples = 2 * num_segments * num_features + 2048

        shap_vals, baseline, boundaries = self.compute_importance(
            x, lengths,
            num_segments=num_segments,
            nsamples=nsamples,
            output_type=output_type,
            target_bin=target_bin,
            mask_value=mask_value,
            show_progress=True,
            importance_type='temporal_feature',
        )
        shap_2d = shap_vals.reshape(num_segments, num_features)  # (S, F)

        # --- layout (identical to visualise_prediction with attention) ----
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1.2])

        # ====== (1) SHAP temporal-feature heatmap (replaces attention) =====
        ax_shap = fig.add_subplot(gs[0, 0])

        # Offset boundaries by start_frame so x-axis matches the feature plot
        abs_boundaries = boundaries + start_frame
        vmax = np.abs(shap_2d).max()
        im = ax_shap.imshow(
            shap_2d.T,                       # (F, S) — features on y-axis
            aspect='auto',
            cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[abs_boundaries[0], abs_boundaries[-1],
                    len(names) - 0.5, -0.5],
        )
        ax_shap.set_yticks(range(len(names)))
        ax_shap.set_yticklabels(names, fontsize=8)
        ax_shap.set_ylabel("Feature")
        ax_shap.set_title("SHAP Temporal-Feature Importance")
        fig.colorbar(im, ax=ax_shap, label="SHAP value", fraction=0.046, pad=0.04)

        # ===== (2) Features =====
        ax_feat = fig.add_subplot(gs[1, 0])

        features = features[:length, :]
        for f_idx in range(features.shape[1]):
            ax_feat.plot(np.arange(start_frame, start_frame+length), features[:, f_idx] / max(features[:, f_idx]), label=(names[f_idx]), alpha=0.7)

        ax_feat.set_xlabel("Time / frames")
        ax_feat.set_ylabel("Features")
        ax_feat.legend(fontsize=8, ncol=2)
        ax_feat.grid(True)

        # ====== PMFs ======
        ax_sd = fig.add_subplot(gs[:, 1])  # span both rows
        bin_edges = to_numpy(bin_edges)
        abs_bin_edges = bin_edges + landmark_frame
        bin_widths = np.diff(abs_bin_edges)
        bin_widths[-1] = bin_widths[-2]  # make last bin same width for visualization

        ax_sd.bar(
            abs_bin_edges[:-1],
            predicted_pmf,
            width=bin_widths,
            align='edge',
            color='tab:blue',
            edgecolor='k',
            alpha=0.5,
            label='Predicted PMF',
        )

        if true_pmf is not None:
            ax_sd.bar(
                abs_bin_edges[:-1],
                true_pmf,
                width=bin_widths,
                align='edge',
                color='tab:red',
                edgecolor='k',
                alpha=0.5,
                label='True PMF',
                )

        abs_event_time = time_to_event + landmark_frame
        if event_indicator == 1:
            ax_sd.axvline(abs_event_time, color='red', linestyle='--', label="Event Time")
        else:
            ax_sd.axvline(abs_event_time, color='orange', linestyle='--', label="Censoring Time")

        ax_sd.set_xlabel("Time / frames")
        ax_sd.set_ylabel("Probability")
        ax_sd.legend()
        ax_sd.grid(True)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
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
