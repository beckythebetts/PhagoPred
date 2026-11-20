from __future__ import annotations
from dataclasses import dataclass, field, fields

from PhagoPred.utils.logger import get_logger
from ..interpret.data_models import BackgroundEnum

LOG = get_logger(__name__)


@dataclass
class SampleSelectionCfg:
    num_samples: int = field(default=100, compare=False)
    min_sequence_len: int | None = None
    min_horizon_cif: float = 0.0


@dataclass
class BackgroundCfg:
    split: str = 'train'
    pool_size: int = 2000
    # VAR order p in 1..var_max_lag, chosen by var_criterion ('aic' or
    # 'bic') with every p scored on the same windows (var.fit_var). Long
    # enough for the longest dependence between features (the chain
    # scenarios' 50-frame lags).
    var_max_lag: int = 100
    var_criterion: str = 'aic'
    # Zero the VAR coefficients with t^2 < log(n) and refit (var._ols): keeps
    # the precision sparse, so a VAR(50) fill isn't ~10x slower.
    var_prune: bool = True
    var_burn_in: int | None = None


@dataclass
class ShapCfg:
    num_temporal_segments: int = 10
    num_draws: int = 512
    # Draws for importance-weighted fills (OBSERVATIONAL_GT with nonlinear
    # dependence between features), whose effective sample size is a fraction
    # of the draws (nonlinear_chain: ~3-70% of 512, lowest for coalitions
    # with most of the window free).
    weighted_draws: int = 2048
    num_groups: int = 16  # draws split into groups for standard errors
    # Coalition budget per game axis ('temporal', 'feature',
    # 'temporal_feature'); None or missing: 2M + 2048 (exact when 2^M - 2 fits).
    # Chosen so coalition-sampling error stays at or below the background
    # noise of B=512 (measured on linear: ~2-9% vs 5-10%).
    budgets: dict[str, int | None] = field(default_factory=lambda: {
        'temporal': 256,
        'feature': None,
        'temporal_feature': 1024,
    })
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
    # No extra min_sequence_len: the dataset's min_length (180) is already
    # the reach of the longest hazard window, so every landmark has its
    # whole hazard input inside the window.
    'Standard': InterpretCfg(
        SampleSelectionCfg(50, min_sequence_len=None),
        BackgroundCfg(),
        ShapCfg(),
    )
}


def _load_dataclass(cls, d: dict):
    """Recursively construct a dataclass from a dict, ignoring init=False fields."""
    LOG.debug(f'Loading dataclass object {cls}')
    init_fields = {f.name for f in fields(cls) if f.init}
    kwargs = {k: v for k, v in d.items() if k in init_fields}
    return cls(**kwargs)


def load_interpret_cfg(d: dict) -> InterpretCfg:
    shap = dict(d['shap'])
    shap['backgrounds'] = [BackgroundEnum(b) for b in shap['backgrounds']]
    gt = d.get('ground_truth')
    return InterpretCfg(
        selection=_load_dataclass(SampleSelectionCfg, d['selection']),
        background=_load_dataclass(BackgroundCfg, d['background']),
        shap=_load_dataclass(ShapCfg, shap),
        ground_truth=None if gt is None else _load_dataclass(
            GroundTruthCfg, gt),
    )
