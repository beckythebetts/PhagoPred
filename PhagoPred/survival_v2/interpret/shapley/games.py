from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from functools import cached_property

import numpy as np


class AxisEnum(str, Enum):
    TEMPORAL = 'temporal'
    FEATURE = 'feature'
    TEMPORAL_FEATURE = 'temporal_feature'


@dataclass(frozen=True)
class Game:
    axis: AxisEnum
    n_frames: int
    feature_names: list[str]
    num_time_segments: int

    def n_segments(self) -> int:
        if self.axis == AxisEnum.FEATURE:
            return 1
        else:
            return min(self.num_time_segments, self.n_frames)

    def segment_boundaries(self):
        return np.linspace(0, self.n_frames, self.n_segments() + 1, dtype=int)

    @cached_property
    def player_map(self) -> np.ndarray:
        """(T, F) int, which player controls which feature/timestep."""
        n_feats = len(self.feature_names)

        seg = np.repeat(np.arange(self.n_segments()),
                        np.diff(self.segment_boundaries()))  # (T,)
        feat = np.arange(len(self.feature_names))

        if self.axis == AxisEnum.TEMPORAL:
            return np.broadcast_to(seg[:, None], (self.n_frames, n_feats))
        elif self.axis == AxisEnum.FEATURE:
            return np.broadcast_to(feat[None, :], (self.n_frames, n_feats))
        return feat[None, :] * self.n_segments() + seg[:, None]

    def n_players(self):
        return int(self.player_map.max()) + 1

    def pinned_mask(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Args
        ----
            Coalitions (N, n_players) bool
        
        Returns
        -------
            (N, T, F) bool
        """
        return np.asarray(coalitions, dtype=bool)[:, self.player_map]

    def shape_output(self, estimator_output: np.ndarray) -> np.ndarray:
        if self.axis == AxisEnum.TEMPORAL_FEATURE:
            return estimator_output.reshape(len(self.feature_names),
                                            self.n_segments(),
                                            *estimator_output.shape[1:])
        return estimator_output
