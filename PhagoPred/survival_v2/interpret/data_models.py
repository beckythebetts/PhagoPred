from __future__ import annotations
from typing import Literal
from enum import Enum
from dataclasses import dataclass, field

import numpy as np
import zlib

from PhagoPred.utils.logger import get_logger

LOGGER = get_logger()


class BackgroundEnum(str, Enum):
    OBSERVATIONAL_VAR = 'observational_var'
    OBSERVATIONAL_GT = 'observational_gt'
    INTERVENTIONAL = 'interventional'


class PredictorEnum(str, Enum):
    GROUND_TRUTH = 'ground_truth'
    MODEL = 'model'


def _h5(kind: Literal['attr', 'dataset'], **kwargs):
    return {'kind': kind, **kwargs}


@dataclass
class SHAPResult:
    predictor: PredictorEnum = field(metadata=_h5('attr', enum=PredictorEnum))
    background: BackgroundEnum = field(
        metadata=_h5('attr', enum=BackgroundEnum))

    temporal: np.ndarray | None = None
    feature: np.ndarray | None = None
    temporal_feature: np.ndarray | None = None

    temporal_stderr: np.ndarray | None = None
    feature_stderr: np.ndarray | None = None
    temporal_feature_stderr: np.ndarray | None = None

    segment_boundaries: np.ndarray | None = field(default=None,
                                                  metadata=_h5('attr'))


@dataclass
class SampleValues:
    features: np.ndarray
    clean_features: np.ndarray | None = None
    base_noise: np.ndarray | None = None
    context_features: np.ndarray | None = None
    horizon_hazard: np.ndarray | None = None


@dataclass
class SampleWithSHAP:
    feature_names: list[str] = field(metadata=_h5('attr'))
    feature_vals: SampleValues
    landmark_frame: int = field(metadata=_h5('attr'))
    event_frame: int | None = field(default=None, metadata=_h5('attr'))
    shap_vals: dict[tuple[PredictorEnum, BackgroundEnum],
                    SHAPResult] = field(default_factory=dict)
    seed: int | None = field(default=None, metadata=_h5('attr'))

    def __post_init__(self):
        if self.seed is None:
            feats = np.ascontiguousarray(self.feature_vals.features,
                                         dtype=np.float32)
            self.seed = zlib.crc32(
                feats.tobytes() +
                int(self.landmark_frame).to_bytes(8, 'little'))

    def _feat_idxs(self, names: list[str]) -> list[int]:
        missing = [n for n in names if n not in self.feature_names]
        if missing:
            raise KeyError(
                f'features {missing} not in sample {self.feature_names}')
        return [self.feature_names.index(n) for n in names]

    def feat_vals(self,
                  feat_names: list[str],
                  clean: bool = False) -> np.ndarray:
        """
        returns
        -------
            feature_vals [T, F]
        """

        vals = self.feature_vals.clean_features if clean else self.feature_vals.features
        if vals is None:
            LOGGER.info(
                'Tried to get clean sample features, but dont exist. Reverting to base features'
            )
            vals = self.feature_vals.features
        T = min(self.landmark_frame, vals.shape[1])
        return vals[self._feat_idxs(feat_names), :T].T  # [T, F]

    def ctx_vals(self, feat_names: list[str], num_frames: int) -> np.ndarray:
        """
        returns
        ------- 
            context_vals [C, F]
        """
        ctx = self.feature_vals.context_features
        if ctx is None or ctx.shape[1] < num_frames:
            raise ValueError(
                f'sample has fewer than {num_frames} context frames')
        return ctx[self._feat_idxs(feat_names),
                   ctx.shape[1] - num_frames:].T  # [C, F]
