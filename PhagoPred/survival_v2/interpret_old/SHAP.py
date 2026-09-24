"""
SHAP (SHapley Additive exPlanations) for survival analysis models.

Provides feature importance and temporal importance analysis using SHAP values.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import json

import shap

from PhagoPred.survival_v2.models.base import SurvivalModel
from PhagoPred.survival_v2.data.dataset import CellDataset, collate_fn


@dataclass
class SHAPResults:
    """Container for SHAP analysis results."""
    shap_values: np.ndarray  # (num_samples, seq_len, num_features) or aggregated
    feature_names: list[str]
    time_bins: Optional[np.ndarray] = None
    baseline_value: Optional[np.ndarray] = None

    # Aggregated importance scores
    feature_importance: Optional[np.ndarray] = None  # (num_features,)
    temporal_importance: Optional[np.ndarray] = None  # (num_time_bins,)
    feature_temporal_importance: Optional[np.ndarray] = None  # (num_time_bins, num_features)


class ModelWrapper(torch.nn.Module):
    """
    Wrapper to make the survival model compatible with SHAP.
    Handles the model's forward interface and output selection.
    """

    def __init__(
        self,
        model: "SurvivalModel",
        output_type: Literal["pmf", "cif", "expected_time", "median_time", "logits"] = "expected_time",
        target_bin: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.output_type = output_type
        self.target_bin = target_bin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a single scalar or vector output per sample.

        Args:
            x: (batch, seq_len, num_features) - padded feature sequences

        Returns:
            Output tensor based on output_type
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Create lengths tensor (assume all valid)
        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=x.device)

        # Get model outputs
        outputs = self.model(x, lengths, return_attention=False)

        # Handle auxiliary outputs (some models return tuple)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        if self.output_type == "logits":
            if self.target_bin is not None:
                return outputs[:, self.target_bin]
            return outputs

        elif self.output_type == "pmf":
            pmf = self.model.predict_pmf(outputs)
            if self.target_bin is not None:
                return pmf[:, self.target_bin].unsqueeze(-1)
            return pmf

        elif self.output_type == "cif":
            cif = self.model.predict_cif(outputs)
            if self.target_bin is not None:
                return cif[:, self.target_bin]
            return cif

        elif self.output_type == "expected_time":
            # print(self.model.predict_expected_time(outputs), self.model.predict_expected_time(outputs).shape)
            return self.model.predict_expected_time(outputs).unsqueeze(-1)

        elif self.output_type == "median_time":
            return self.model.predict_median_time(outputs)

        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class SurvivalSHAP:
    """
    SHAP analysis for survival models with temporal features.

    Computes SHAP values for:
    - Feature importance (averaged across time)
    - Temporal importance (averaged across features)
    - Feature-temporal interaction importance
    """

    def __init__(
        self,
        model: "SurvivalModel",
        background_data: Union[torch.Tensor, np.ndarray],
        feature_names: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained SurvivalModel
            background_data: Background/reference data for SHAP (batch, seq_len, features)
            feature_names: Names of features for visualization
            device: Device to run computations on
        """
        if shap is None:
            raise ImportError("shap package not installed. Install with: pip install shap")

        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Convert background data to tensor
        if isinstance(background_data, np.ndarray):
            background_data = torch.tensor(background_data, dtype=torch.float32)
        self.background_data = background_data.to(device)

        # Store dimensions
        self.num_background = background_data.shape[0]
        self.seq_len = background_data.shape[1]
        self.num_features = background_data.shape[2]

        # Feature names
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(self.num_features)]
        else:
            self.feature_names = feature_names

        self._explainer_cache = {}

    def _get_explainer(
        self,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        method: Literal["deep", "gradient", "kernel"] = "gradient",
    ) -> shap.Explainer:
        """Get or create SHAP explainer for given output configuration."""
        cache_key = (output_type, target_bin, method)

        if cache_key not in self._explainer_cache:
            wrapped_model = ModelWrapper(self.model, output_type, target_bin)
            wrapped_model.to(self.device)
            wrapped_model.eval()

            if method == "deep":
                explainer = shap.DeepExplainer(wrapped_model, self.background_data)
            elif method == "gradient":
                explainer = shap.GradientExplainer(wrapped_model, self.background_data)
            elif method == "kernel":
                # For kernel, we need a function not a module
                def model_fn(x):
                    with torch.no_grad():
                        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
                        out = wrapped_model(x_tensor)
                        return out.cpu().numpy()
                explainer = shap.KernelExplainer(model_fn, self.background_data.cpu().numpy())
            else:
                raise ValueError(f"Unknown method: {method}")

            self._explainer_cache[cache_key] = explainer

        return self._explainer_cache[cache_key]

    def compute_shap_values(
        self,
        data: Union[torch.Tensor, np.ndarray],
        output_type: Literal["pmf", "cif", "expected_time", "median_time", "logits"] = "expected_time",
        target_bin: Optional[int] = None,
        method: Literal["deep", "gradient", "kernel"] = "gradient",
    ) -> np.ndarray:
        """
        Compute raw SHAP values for input data.

        Args:
            data: Input data (batch, seq_len, num_features)
            output_type: Which model output to explain
            target_bin: If output_type returns multi-bin output, which bin to explain
            method: SHAP method to use

        Returns:
            SHAP values array (batch, seq_len, num_features)
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)

        explainer = self._get_explainer(output_type, target_bin, method)
        if method == "kernel":
            print(data.shape)
            shap_values = explainer.shap_values(data.cpu().numpy())
        else:
            shap_values = explainer.shap_values(data)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()

        shap_values = np.squeeze(shap_values)
        return shap_values

    def compute_feature_importance(
        self,
        data: Union[torch.Tensor, np.ndarray],
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        method: str = "gradient",
        aggregation: Literal["mean_abs", "mean", "sum_abs"] = "mean_abs",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute feature importance by aggregating SHAP values across time.

        Args:
            data: Input data (batch, seq_len, num_features)
            output_type: Which model output to explain
            target_bin: If output_type is multi-bin, which bin to explain
            method: SHAP method
            aggregation: How to aggregate across time and samples

        Returns:
            (importance_scores, shap_values) - importance per feature, raw SHAP values
        """
        shap_values = self.compute_shap_values(data, output_type, target_bin, method)

        # Aggregate across time dimension first, then samples
        if aggregation == "mean_abs":
            # Mean absolute SHAP value per feature
            importance = np.abs(shap_values).mean(axis=(0, 1))
        elif aggregation == "mean":
            importance = shap_values.mean(axis=(0, 1))
        elif aggregation == "sum_abs":
            importance = np.abs(shap_values).sum(axis=(0, 1))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return importance, shap_values

    def compute_temporal_importance(
        self,
        data: Union[torch.Tensor, np.ndarray],
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        method: str = "gradient",
        aggregation: Literal["mean_abs", "mean", "sum_abs"] = "mean_abs",
        num_time_bins: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute temporal importance by aggregating SHAP values across features.

        Args:
            data: Input data (batch, seq_len, num_features)
            output_type: Which model output to explain
            target_bin: If output_type is multi-bin, which bin to explain
            method: SHAP method
            aggregation: How to aggregate across features and samples
            num_time_bins: If set, group timesteps into this many bins

        Returns:
            (importance_scores, shap_values) - importance per time step/bin
        """
        shap_values = self.compute_shap_values(data, output_type, target_bin, method)
        seq_len = shap_values.shape[1]

        # Aggregate across features first
        if aggregation == "mean_abs":
            time_shap = np.abs(shap_values).mean(axis=2)  # (batch, seq_len)
        elif aggregation == "mean":
            time_shap = shap_values.mean(axis=2)
        elif aggregation == "sum_abs":
            time_shap = np.abs(shap_values).sum(axis=2)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Optionally bin timesteps
        if num_time_bins is not None and num_time_bins < seq_len:
            bin_edges = np.linspace(0, seq_len, num_time_bins + 1, dtype=int)
            binned_importance = np.zeros((time_shap.shape[0], num_time_bins))
            for i in range(num_time_bins):
                binned_importance[:, i] = time_shap[:, bin_edges[i]:bin_edges[i+1]].mean(axis=1)
            time_shap = binned_importance

        # Average across samples
        importance = time_shap.mean(axis=0)

        return importance, shap_values

    def compute_feature_temporal_importance(
        self,
        data: Union[torch.Tensor, np.ndarray],
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        method: str = "gradient",
        num_time_bins: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute importance for each (time_bin, feature) combination.

        Args:
            data: Input data (batch, seq_len, num_features)
            output_type: Which model output to explain
            target_bin: If output_type is multi-bin, which bin to explain
            method: SHAP method
            num_time_bins: Number of temporal bins to group timesteps into

        Returns:
            (importance_matrix, shap_values) - shape (num_time_bins, num_features)
        """
        shap_values = self.compute_shap_values(data, output_type, target_bin, method)
        seq_len = shap_values.shape[1]

        if num_time_bins is None:
            num_time_bins = seq_len

        # Bin timesteps
        if num_time_bins < seq_len:
            bin_edges = np.linspace(0, seq_len, num_time_bins + 1, dtype=int)
            binned_shap = np.zeros((shap_values.shape[0], num_time_bins, self.num_features))
            for i in range(num_time_bins):
                binned_shap[:, i, :] = shap_values[:, bin_edges[i]:bin_edges[i+1], :].mean(axis=1)
            shap_for_matrix = binned_shap
        else:
            shap_for_matrix = shap_values

        # Mean absolute value across samples
        importance_matrix = np.abs(shap_for_matrix).mean(axis=0)

        return importance_matrix, shap_values

    def analyse(
        self,
        data: Union[torch.Tensor, np.ndarray, 'CellDataset'],
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        method: str = "gradient",
        num_time_bins: Optional[int] = None,
        num_samples: Optional[int] = 100,
    ) -> SHAPResults:
        """
        Full SHAP analysis returning all importance metrics.

        Args:
            data: Input data to explain. If CellDataset, samples will be drawn.
            output_type: Which model output to explain
            target_bin: Target bin for multi-bin outputs
            method: SHAP method
            num_time_bins: Number of temporal bins

        Returns:
            SHAPResults dataclass with all computed importance values
        """
        if isinstance(data, CellDataset):
            data = sample_cell_dataset(data, num_samples=num_samples)
            
        # Compute raw SHAP values
        shap_values = self.compute_shap_values(data, output_type, target_bin, method)

        # Feature importance
        feature_importance = np.abs(shap_values).mean(axis=(0, 1))

        # Temporal importance
        temporal_shap = np.abs(shap_values).mean(axis=2)  # (batch, seq_len)
        seq_len = shap_values.shape[1]

        if num_time_bins is not None and num_time_bins < seq_len:
            bin_edges = np.linspace(0, seq_len, num_time_bins + 1, dtype=int)
            binned_temporal = np.zeros((temporal_shap.shape[0], num_time_bins))
            for i in range(num_time_bins):
                binned_temporal[:, i] = temporal_shap[:, bin_edges[i]:bin_edges[i+1]].mean(axis=1)
            temporal_importance = binned_temporal.mean(axis=0)
            time_bins = bin_edges
        else:
            temporal_importance = temporal_shap.mean(axis=0)
            time_bins = np.arange(seq_len + 1)
            num_time_bins = seq_len

        # Feature-temporal matrix
        if num_time_bins < seq_len:
            bin_edges = np.linspace(0, seq_len, num_time_bins + 1, dtype=int)
            binned_ft = np.zeros((shap_values.shape[0], num_time_bins, self.num_features))
            for i in range(num_time_bins):
                binned_ft[:, i, :] = shap_values[:, bin_edges[i]:bin_edges[i+1], :].mean(axis=1)
            ft_importance = np.abs(binned_ft).mean(axis=0)
        else:
            ft_importance = np.abs(shap_values).mean(axis=0)

        return SHAPResults(
            shap_values=shap_values,
            feature_names=self.feature_names,
            time_bins=time_bins,
            feature_importance=feature_importance,
            temporal_importance=temporal_importance,
            feature_temporal_importance=ft_importance,
        )
    
    def analyse_all_bins(
        self,
        data: Union[torch.Tensor, np.ndarray, 'CellDataset'],
        method: str = "gradient",
        num_time_bins: Optional[int] = None,
        num_samples: Optional[int] = 100,
    ) -> list[SHAPResults]:
        """
        Run SHAP analysis for each target time bin separately.

        Args:
            data: Input data to explain
            method: SHAP method
            num_input_time_bins: Number of bins to group input timesteps into
            num_samples: Number of samples if data is CellDataset

        Returns:
            List of SHAPResults, one per target bin
        """
        num_target_bins = self.model.num_bins

        # Sample data once to use for all bins
        if isinstance(data, CellDataset):
            data = sample_cell_dataset(data, num_samples=num_samples)

        results_per_bin = []
        for bin_idx in tqdm(range(num_target_bins)):
            result = self.analyse(
                data,
                output_type="pmf",
                target_bin=bin_idx,
                method=method,
                num_time_bins=num_time_bins,
                num_samples=num_samples,
            )
            results_per_bin.append(result)
        return results_per_bin

    def plot_feature_importance_per_bin(
        self,
        results_per_bin: list[SHAPResults],
        bin_labels: Optional[list[str]] = None,
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot heatmap of feature importance across target bins.

        Args:
            results_per_bin: List of SHAPResults from analyse_all_bins
            bin_labels: Optional labels for target bins
            figsize: Figure size
            save_path: Path to save figure
        """
        num_bins = len(results_per_bin)
        num_features = len(self.feature_names)

        # Build importance matrix: (num_target_bins, num_features)
        importance_matrix = np.zeros((num_bins, num_features))
        for i, result in enumerate(results_per_bin):
            importance_matrix[i] = result.feature_importance

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')

        ax.set_xlabel("Target Time Bin")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance per Target Bin (SHAP)")

        # Y-axis: features
        ax.set_yticks(range(num_features))
        ax.set_yticklabels(self.feature_names)

        # X-axis: target bins
        if bin_labels is None:
            bin_labels = [f"Bin {i}" for i in range(num_bins)]
        ax.set_xticks(range(num_bins))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean |SHAP value|")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_feature_importance_lines(
        self,
        results_per_bin: list[SHAPResults],
        bin_labels: Optional[list[str]] = None,
        top_k: int = 5,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot line chart showing how feature importance changes across target bins.

        Args:
            results_per_bin: List of SHAPResults from analyse_all_bins
            bin_labels: Optional labels for target bins
            top_k: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure
        """
        num_bins = len(results_per_bin)
        num_features = len(self.feature_names)

        # Build importance matrix
        importance_matrix = np.zeros((num_bins, num_features))
        for i, result in enumerate(results_per_bin):
            importance_matrix[i] = result.feature_importance

        # Get top-k features by average importance
        avg_importance = importance_matrix.mean(axis=0)
        top_indices = np.argsort(avg_importance)[::-1][:top_k]

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(num_bins)
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))

        for idx, feat_idx in enumerate(top_indices):
            ax.plot(x, importance_matrix[:, feat_idx], 'o-',
                   color=colors[idx], label=self.feature_names[feat_idx],
                   linewidth=2, markersize=6)

        if bin_labels is None:
            bin_labels = [f"Bin {i}" for i in range(num_bins)]
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')

        ax.set_xlabel("Target Time Bin")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_k} Feature Importance Across Target Bins")
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_all_bins_summary(
        self,
        results_per_bin: list[SHAPResults],
        bin_labels: Optional[list[str]] = None,
        figsize: tuple = (18, 10),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Comprehensive summary plot for multi-bin analysis.

        Args:
            results_per_bin: List of SHAPResults from analyse_all_bins
            bin_labels: Optional labels for target bins
            figsize: Figure size
            save_path: Path to save figure
        """
        num_bins = len(results_per_bin)
        num_features = len(self.feature_names)

        # Build importance matrix
        importance_matrix = np.zeros((num_bins, num_features))
        for i, result in enumerate(results_per_bin):
            importance_matrix[i] = result.feature_importance

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        if bin_labels is None:
            bin_labels = [f"Bin {i}" for i in range(num_bins)]

        # Top left: Heatmap of feature importance per bin
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')
        ax1.set_xlabel("Target Time Bin")
        ax1.set_ylabel("Feature")
        ax1.set_title("Feature Importance per Target Bin")
        ax1.set_yticks(range(num_features))
        ax1.set_yticklabels(self.feature_names)
        ax1.set_xticks(range(num_bins))
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        # Top right: Average feature importance across all bins
        ax2 = fig.add_subplot(gs[0, 1])
        avg_importance = importance_matrix.mean(axis=0)
        sorted_idx = np.argsort(avg_importance)[::-1]
        sorted_imp = avg_importance[sorted_idx]
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        ax2.barh(range(num_features), sorted_imp[::-1], color='steelblue')
        ax2.set_yticks(range(num_features))
        ax2.set_yticklabels(sorted_names[::-1])
        ax2.set_xlabel("Mean |SHAP value|")
        ax2.set_title("Average Feature Importance (All Bins)")
        ax2.grid(axis='x', alpha=0.3)

        # Bottom left: Line plot of top features across bins
        ax3 = fig.add_subplot(gs[1, 0])
        top_k = min(5, num_features)
        top_indices = np.argsort(avg_importance)[::-1][:top_k]
        colors = plt.cm.tab10(np.linspace(0, 1, top_k))
        x = np.arange(num_bins)
        for idx, feat_idx in enumerate(top_indices):
            ax3.plot(x, importance_matrix[:, feat_idx], 'o-',
                    color=colors[idx], label=self.feature_names[feat_idx],
                    linewidth=2, markersize=5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax3.set_xlabel("Target Time Bin")
        ax3.set_ylabel("Mean |SHAP value|")
        ax3.set_title(f"Top {top_k} Features Across Bins")
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(alpha=0.3)

        # Bottom right: Total importance per bin (stacked by feature)
        ax4 = fig.add_subplot(gs[1, 1])
        total_per_bin = importance_matrix.sum(axis=1)
        ax4.bar(x, total_per_bin, color='coral', edgecolor='darkred')
        ax4.set_xticks(x)
        ax4.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax4.set_xlabel("Target Time Bin")
        ax4.set_ylabel("Total |SHAP value|")
        ax4.set_title("Total Feature Importance per Bin")
        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle("SHAP Analysis Across All Target Bins", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    # ==================== Visualization Methods ====================

    def plot_feature_importance(
        self,
        importance: np.ndarray,
        title: str = "Feature Importance (SHAP)",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot bar chart of feature importance."""
        fig, ax = plt.subplots(figsize=figsize)

        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_names = [self.feature_names[i] for i in sorted_idx]

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

    def plot_temporal_importance(
        self,
        importance: np.ndarray,
        time_bins: Optional[np.ndarray] = None,
        title: str = "Temporal Importance (SHAP)",
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot temporal importance as line/bar chart."""
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(importance))
        if time_bins is not None and len(time_bins) == len(importance) + 1:
            # Use bin centers for x-axis
            x = (time_bins[:-1] + time_bins[1:]) / 2

        ax.bar(x, importance, alpha=0.7, color='steelblue', edgecolor='navy')
        ax.plot(x, importance, 'o-', color='darkblue', markersize=4)

        ax.set_xlabel("Time step / bin")
        ax.set_ylabel("Mean |SHAP value|")
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_feature_temporal_heatmap(
        self,
        importance_matrix: np.ndarray,
        time_bins: Optional[np.ndarray] = None,
        title: str = "Feature-Temporal Importance (SHAP)",
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot heatmap of feature importance across time bins."""
        fig, ax = plt.subplots(figsize=figsize)

        # importance_matrix shape: (num_time_bins, num_features)
        im = ax.imshow(importance_matrix.T, aspect='auto', cmap='YlOrRd')

        ax.set_xlabel("Time bin")
        ax.set_ylabel("Feature")
        ax.set_title(title)

        # Y-axis labels (features)
        ax.set_yticks(range(len(self.feature_names)))
        ax.set_yticklabels(self.feature_names)

        # X-axis labels (time bins)
        if time_bins is not None and len(time_bins) == importance_matrix.shape[0] + 1:
            bin_labels = [f"{time_bins[i]:.0f}-{time_bins[i+1]:.0f}"
                         for i in range(len(time_bins)-1)]
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean |SHAP value|")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_summary(
        self,
        results: SHAPResults,
        figsize: tuple = (16, 12),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot comprehensive summary of SHAP analysis."""
        fig = plt.figure(figsize=figsize)

        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Feature importance (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        sorted_idx = np.argsort(results.feature_importance)[::-1]
        sorted_imp = results.feature_importance[sorted_idx]
        sorted_names = [self.feature_names[i] for i in sorted_idx]
        ax1.barh(range(len(sorted_imp)), sorted_imp[::-1], color='steelblue')
        ax1.set_yticks(range(len(sorted_imp)))
        ax1.set_yticklabels(sorted_names[::-1])
        ax1.set_xlabel("Mean |SHAP value|")
        ax1.set_title("Feature Importance")
        ax1.grid(axis='x', alpha=0.3)

        # Temporal importance (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        x = np.arange(len(results.temporal_importance))
        ax2.bar(x, results.temporal_importance, color='coral', alpha=0.7)
        ax2.plot(x, results.temporal_importance, 'o-', color='darkred', markersize=3)
        ax2.set_xlabel("Time bin")
        ax2.set_ylabel("Mean |SHAP value|")
        ax2.set_title("Temporal Importance")
        ax2.grid(axis='y', alpha=0.3)

        # Feature-temporal heatmap (bottom, spanning both columns)
        ax3 = fig.add_subplot(gs[1, :])
        im = ax3.imshow(results.feature_temporal_importance.T, aspect='auto', cmap='YlOrRd')
        ax3.set_xlabel("Time bin")
        ax3.set_ylabel("Feature")
        ax3.set_title("Feature-Temporal Importance")
        ax3.set_yticks(range(len(self.feature_names)))
        ax3.set_yticklabels(self.feature_names)
        cbar = plt.colorbar(im, ax=ax3, orientation='vertical', shrink=0.8)
        cbar.set_label("Mean |SHAP value|")

        plt.suptitle("SHAP Analysis Summary", fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

def sample_cell_dataset(dataset: "CellDataset", num_samples: int = 100, max_len: int = 1000) -> torch.Tensor:
    """
    Sample random sequences from CellDataset..

    Args:
        dataset: CellDataset to sample from
        num_samples: Number of samples to draw
    Returns:
        Tensor of shape (num_samples, seq_len, num_features)
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, dataset, max_seq_len=max_len),
    )
    batch = next(iter(dataloader))
    return batch['features']
    
def create_background_from_dataset(
    dataset: "CellDataset",
    num_samples: int = 100,
    sample_type: Literal['sample', 'mean'] = 'sample',
    max_len: int = 1000,
) -> torch.Tensor:
    """
    Create background data for SHAP from CellDataset.

    Args:
        dataset: CellDataset to sample from
        num_samples: Number of samples to draw
        sample_type: 'sample' or 'mean'
        """
    bg_dataset = sample_cell_dataset(dataset, num_samples, max_len)
    if sample_type == 'mean':
        bg_dataset = torch.mean(bg_dataset, dim=0, keepdim=True)
    
    return bg_dataset

def explain_single_sample(
    model: "SurvivalModel",
    sample: Union[torch.Tensor, np.ndarray],
    background: torch.Tensor,
    feature_names: list[str],
    output_type: str = "expected_time",
    method: str = "gradient",
    device: str = "cpu",
) -> SHAPResults:
    """
    Convenience function to explain a single sample.

    Args:
        model: Trained survival model
        sample: Single sample (seq_len, num_features) or (1, seq_len, num_features)
        background: Background data for SHAP
        feature_names: Feature names
        output_type: Model output to explain
        method: SHAP method
        device: Computation device

    Returns:
        SHAPResults for the single sample
    """
    explainer = SurvivalSHAP(
        model=model,
        background_data=background,
        feature_names=feature_names,
        device=device,
    )

    if isinstance(sample, np.ndarray):
        sample = torch.tensor(sample, dtype=torch.float32)
    if sample.dim() == 2:
        sample = sample.unsqueeze(0)

    # Ensure sample has same seq_len as background
    bg_seq_len = background.shape[1]
    sample_seq_len = sample.shape[1]

    if sample_seq_len != bg_seq_len:
        # Pad or truncate
        if sample_seq_len > bg_seq_len:
            sample = sample[:, -bg_seq_len:]
        else:
            padded = torch.zeros(1, bg_seq_len, sample.shape[2])
            padded[:, -sample_seq_len:] = sample
            sample = padded

    return explainer.analyse(sample, output_type=output_type, method=method)


# # Alias for backward compatibility
# SHAP = SurvivalSHAP


        