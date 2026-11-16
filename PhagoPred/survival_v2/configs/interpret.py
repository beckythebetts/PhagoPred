from __future__ import annotations
from dataclasses import dataclass, field

from ..interpret.data_models import BackgroundEnum


@dataclass
class SampleSelectionCfg:
    num_samples: int = field(default=100, compare=False)
    min_sequence_len: int | None = None
    min_horizon_cif: float = 0.0


@dataclass
class BackgroundCfg:
    split: str = 'train'
    pool_size: int = 2000
    var_max_lag: int = 5
    var_burn_in: int | None = None


@dataclass
class ShapCfg:
    num_temporal_segments: int = 10
    num_draws: int = 512
    num_shap_samples: int | None = None
    seed: int = 42
    backgrounds: list[BackgroundEnum] = field(default_factory=lambda: [
        BackgroundEnum.INTERVENTIONAL,
        BackgroundEnum.OBSERVATIONAL_GT,
        BackgroundEnum.OBSERVATIONAL_VAR,
    ])


@dataclass
class GroundTruthCfg:
    saved_burn_in_frames: int | None = None


@dataclass
class InterpretCfg:
    selection: SampleSelectionCfg
    background: BackgroundCfg
    shap: ShapCfg
    ground_truth: GroundTruthCfg | None = None


INTERPRETS = {
    'Standard': InterpretCfg(
        SampleSelectionCfg(100),
        BackgroundCfg(),
        ShapCfg(),
    )
}
