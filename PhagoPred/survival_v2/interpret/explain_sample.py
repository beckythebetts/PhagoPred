# interpret/explain.py
from __future__ import annotations
from dataclasses import dataclass, field

from .data_models import SampleWithSHAP, SHAPResult, PredictorEnum, BackgroundEnum
from .shapley.games import Game, AxisEnum
from .shapley.explainer import kernel_shap
from .value_func import ValueFunction
from .utils.seeds import get_generator, COALITIONS


@dataclass
class ExplainCfg:
    num_time_segments: int = 10
    # Coalition budget per axis value; None or missing: kernel_shap's default.
    budgets: dict[str, int | None] = field(default_factory=dict)
    axes: tuple[AxisEnum, ...] = tuple(AxisEnum)


def explain_sample(sample: SampleWithSHAP, vf: ValueFunction,
                   players: list[str], predictor: PredictorEnum,
                   background: BackgroundEnum, cfg: ExplainCfg) -> SHAPResult:
    vf.set_sample(sample)
    result = SHAPResult(predictor, background)
    for axis_idx, axis in enumerate(cfg.axes):
        game = Game(axis, sample.landmark_frame, players,
                    cfg.num_time_segments)
        est = kernel_shap(vf.for_game(game),
                          game.n_players(),
                          cfg.budgets.get(axis.value),
                          rng=get_generator(sample, COALITIONS, axis_idx))
        setattr(result, axis.value, game.shape_output(est.shap_values))
        if est.stderr is not None:
            setattr(result, f'{axis.value}_stderr',
                    game.shape_output(est.stderr))
        if axis != AxisEnum.FEATURE:
            result.segment_boundaries = game.segment_boundaries()
    return result
