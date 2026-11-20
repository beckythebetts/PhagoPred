from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ScenarioCfg
from PhagoPred.survival_v2.data.graph_synthetic.dynamics import draw_noise, extend_signals
from PhagoPred.survival_v2.data.graph_synthetic.hazard import pmf_from_hazards, bin_hazards
from PhagoPred.survival_v2.utils.io import load_model
from PhagoPred.survival_v2.models import SurvivalModel
from PhagoPred.survival_v2.configs import ExperimentCfg, PredictionTarget
from .bg_fill import FillResult
from .targets import rmst, event_probability
from ..data_models import SampleWithSHAP
from ..utils.seeds import get_generator, PREDICTOR
"""
N = number of coalitions
B = number of background_draws
"""


class Predictor(ABC):

    @abstractmethod
    def __call__(self, fill: FillResult) -> np.ndarray:
        ...  # [N, B]

    def set_sample(self, sample: SampleWithSHAP) -> None:
        """Default: nothing per-sample (e.g. a deterministic model)."""


class GraphPredictor(Predictor):

    def __init__(
        self,
        scenario: ScenarioCfg,
        target: PredictionTarget,
        player_features: list[str],
        num_draws: int,
    ):
        self.scenario = scenario
        self.graph = scenario.graph
        self.horizon = (target.horizon
                        if target.dataset_type in ('binary', 'hazard') else
                        int(target.bins[-1]))
        self.target = target
        self.player_features = player_features
        self.all_features = [f.name for f in self.graph.features]

        self.num_draws = num_draws
        self.window_rules = [
            r for r in self.graph.rules if r.target not in self.player_features
        ]

        self.sample = None
        self.rng = None
        self.future_noise = None

    def __call__(self, fill: FillResult) -> np.ndarray:  # [N, B]
        N, B, T, F = fill.values.shape
        c = self.graph.max_lag
        if B != self.num_draws:
            raise ValueError(
                f'fill has {B} draws, predictor expects {self.num_draws}')

        ctx = {
            k: np.repeat(v[:, None], N * B, axis=1)
            for k, v in self.ctx.items()
        }  # [c, N*B]

        if fill.context is not None:
            for i, p in enumerate(self.player_features):
                ctx[p] = fill.context[:, :, -c:, i].reshape(N * B, c).T

        window_values = {
            k: np.tile(v, (1, N))
            for k, v in self.window_noise.items()
        }
        future_values = {
            k: np.tile(v, (1, N))
            for k, v in self.future_noise.items()
        }

        for i, feat_name in enumerate(self.player_features):
            window_values[feat_name] = fill.values[..., i].reshape(N * B, T).T

        values = extend_signals(self.graph,
                                ctx,
                                window_values,
                                frame_0=c,
                                rules=self.window_rules,
                                return_full=True)

        future = extend_signals(self.graph,
                                values,
                                future_values,
                                frame_0=c,
                                return_full=False)

        if self.target.dataset_type == 'hazard':
            # Mean scale * Hazard over frames L + 1 .. L + H (future index i
            # is frame L + i), as BinaryCellDataset's 'hazard' label.
            scale = self.scenario.hazard_func.params.get('scale', 1.0)
            z = future['Hazard'][1:self.horizon + 1]
            return (scale * z).mean(axis=0).reshape(N, B)
        h = self.scenario.hazard_func(future['Hazard'])
        h[max(self.scenario.num_frames - self.sample.landmark_frame, 0):] = 0.0
        return self._target(h).reshape(N, B)

    def set_sample(self, sample: SampleWithSHAP) -> None:
        self.sample = sample
        self.rng = get_generator(sample, PREDICTOR)
        self.future_noise = draw_noise(self.graph, self.horizon + 1,
                                       self.num_draws, self.rng)
        self.window_noise = draw_noise(
            self.graph,
            self.sample.landmark_frame,
            self.num_draws,
            self.rng,
        )

        ctx = sample.ctx_vals(self.all_features, self.graph.max_lag)
        self.ctx = {
            name: ctx[:, j]
            for j, name in enumerate(self.all_features)
        }

    def _target(self, hazard: np.ndarray) -> np.ndarray:  # hazard [H+1, n]
        if self.target.dataset_type == 'binary':
            return event_probability(hazard, self.horizon)
        if self.target.dataset_type == 'survival':
            bins = np.asarray(self.target.bins)
            pmf = pmf_from_hazards(bin_hazards(hazard, bins))  # [num_bins, n]
            return rmst(pmf.T, bins)
        raise ValueError(self.target.dataset_type)


class ModelPredictor(Predictor):

    def __init__(self,
                 model: SurvivalModel,
                 cfg: ExperimentCfg,
                 max_batch_size: int = 2048):
        self.model = model
        self.target = cfg.dataset.prediction_target
        self.player_features = list(cfg.feature_combo)
        self.means = np.asarray(cfg.dataset.normalisation_means,
                                dtype=np.float32)
        self.stds = np.asarray(cfg.dataset.normalisation_stds,
                               dtype=np.float32)
        self.max_batch_size = max_batch_size

    @classmethod
    def from_experiment_dir(cls, experiment_dir: Path):
        model, cfg = load_model(experiment_dir)
        return cls(model, cfg)

    def __call__(self, fill: FillResult) -> np.ndarray:
        x = fill.values
        N, B, T, F = x.shape
        x = ((x - self.means) / self.stds).reshape(N * B, T,
                                                   F).astype(np.float32)
        out = []
        for start in range(0, len(x), self.max_batch_size):
            chunk = torch.as_tensor(x[start:min(start +
                                                self.max_batch_size, len(x))],
                                    device=self.model.device)
            lengths = torch.full((len(chunk), ), T, device=self.model.device)
            with torch.no_grad():
                logits = self.model(chunk, lengths, return_attention=False)
                if isinstance(logits, tuple):
                    logits = logits[0]
                out.append(self._target(logits).reshape(-1))
        return np.concatenate(out).reshape(N, B)

    def _target(self, logits) -> np.ndarray:
        if self.target.dataset_type == 'hazard':
            return logits.reshape(-1).cpu().numpy()  # regression: raw output
        if self.target.dataset_type == 'binary':
            return self.model.predict_binary(logits).reshape(-1).cpu().numpy()
        if self.target.dataset_type == 'survival':
            pmf = self.model.predict_pmf(logits).cpu().numpy()  # [n, num_bins]
            return rmst(pmf, np.asarray(self.target.bins))
        raise ValueError(self.target.dataset_type)
