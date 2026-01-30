from .SHAP import (
    SurvivalSHAP,
    SHAPResults,
    ModelWrapper,
    create_background_from_dataset,
    explain_single_sample,
)

from .SHAP_kernel import (
    KernelSHAP,
    KernelSHAPResults,
    TemporalMaskWrapper,
    FeatureMaskWrapper,
    quick_temporal_importance,
    quick_feature_importance,
)

__all__ = [
    # Gradient-based SHAP
    "SurvivalSHAP",
    "SHAPResults",
    "ModelWrapper",
    "create_background_from_dataset",
    "explain_single_sample",
    # Kernel/perturbation-based SHAP
    "KernelSHAP",
    "KernelSHAPResults",
    "TemporalMaskWrapper",
    "FeatureMaskWrapper",
    "quick_temporal_importance",
    "quick_feature_importance",
]
