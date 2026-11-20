from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..data_models import SampleWithSHAP
from ..shapley.games import Game
from .bg_fill import BGFill
from .predictor import Predictor
"""
N = number of coalitions
B = number of background draws
R = number of groups (B split into R equal, contiguous slices, for standard errors)
"""


@dataclass
class ValueFunction:
    bg_fill: BGFill
    predictor: Predictor
    num_groups: int
    # N * B * T * F floats per chunk. GraphPredictor holds several arrays of
    # this size at once: 2^27 peaked at 8 GB (OOM on a 15 GB machine); 2^22
    # peaks under 1 GB for every fill, at the same speed (B=512, L~270).
    max_batch_elements: int = 2**22

    def __post_init__(self):
        B = self.bg_fill.num_draws
        if B % self.num_groups:
            raise ValueError(
                f'num_draws {B} not divisible by num_groups {self.num_groups}')
        p_draws = getattr(self.predictor, 'num_draws', None)
        if p_draws is not None and p_draws != B:
            raise ValueError('bg_fill and predictor disagree on num_draws')
        self.min_ess = np.inf

    def __call__(self, pinned: np.ndarray) -> np.ndarray:
        """
        pinned [N, T, F]
        """
        N, T, F = pinned.shape
        B = self.bg_fill.num_draws
        chunk = max(1, self.max_batch_elements // (B * T * F))
        return np.concatenate(
            [self._evaluate(pinned[i:i + chunk]) for i in range(0, N, chunk)])

    def _evaluate(self, pinned: np.ndarray) -> np.ndarray:
        fill = self.bg_fill(pinned)
        values = self.predictor(fill)
        N, B = values.shape
        R = self.num_groups

        if fill.log_weights is None:
            pooled = values.mean(1)
            groups = values.reshape(N, R, B // R).mean(2)
        else:
            pooled = _weighted_mean(values, fill.log_weights)
            groups = _weighted_mean(values.reshape(N, R, B // R),
                                    fill.log_weights.reshape(N, R, B // R))
            self.min_ess = min(self.min_ess, _ess(fill.log_weights).min())
        return np.concatenate([pooled[:, None], groups], axis=1)

    def set_sample(self, sample: SampleWithSHAP) -> None:
        self.bg_fill.set_sample(sample)
        self.predictor.set_sample(sample)
        self.min_ess = np.inf

    def for_game(self, game: Game) -> callable:
        return lambda coalitions: self(game.pinned_mask(coalitions))


def _normalised_weights(log_w: np.ndarray) -> np.ndarray:
    """Self-normalised importance weights over the last axis."""
    w = np.exp(log_w - log_w.max(-1, keepdims=True))
    return w / w.sum(-1, keepdims=True)


def _weighted_mean(values: np.ndarray, log_w: np.ndarray) -> np.ndarray:
    return (_normalised_weights(log_w) * values).sum(-1)


def _ess(log_w: np.ndarray) -> np.ndarray:  #effective smaple size
    """Effective sample size per coalition, (sum w)^2 / sum w^2."""
    w = _normalised_weights(log_w)
    return 1.0 / (w**2).sum(-1)
