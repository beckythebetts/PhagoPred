from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ...utils.precision import ObservedPrecision, condition_and_sample
from ..data_models import SampleWithSHAP
from ..utils.seeds import get_generator, FILL
"""
N = number of coalitions
B = number of background draws
T = num times teps
C = num context time steps
F = num features
"""


@dataclass
class FillResult:
    values: np.ndarray  # [N, B, T, F]
    context: np.ndarray | None = None  # [N, B, C, F]
    log_weights: np.ndarray | None = None  # [N, B] importance weights from ObservationalGT in nonlinear case


class BGFill(ABC):

    def __init__(self, feature_names: list[str], num_draws: int, **kwargs):
        self.feature_names = feature_names
        self.num_draws = num_draws

        self.base_vals = None
        self.rng = None

    def set_sample(self, sample: SampleWithSHAP) -> None:
        self.base_vals = sample.feat_vals(self.feature_names, clean=True)
        self.rng = get_generator(sample, FILL)

    @abstractmethod
    def __call__(self, pinned: np.ndarray) -> FillResult:
        pass


class Interventional(BGFill):

    def __init__(self, feature_names, num_draws, bg_samples: np.ndarray):
        super().__init__(feature_names, num_draws)
        self.bg_samples = bg_samples  # [n_samples, T, F]
        self.current_bg_samples = None

    def set_sample(self, sample: SampleWithSHAP) -> None:
        super().set_sample(sample)
        sample_idxs = self.rng.choice(len(self.bg_samples),
                                      self.num_draws,
                                      replace=False)
        self.current_bg_samples = self.bg_samples[sample_idxs]

    def __call__(self, pinned):
        values = np.where(pinned[:, None], self.base_vals[None, None],
                          self.current_bg_samples[None])

        return FillResult(values)


class Observational(BGFill, ABC):

    def __init__(self, feature_names, num_draws, prec: ObservedPrecision,
                 n_context: int, **kwargs):
        super().__init__(feature_names, num_draws)
        self.prec, self.c = prec, n_context

        self.flat_base = None  # vecotr of base_vals (including context)
        self.xi = None  # Gaussian draw for sampling from self.prec

    def set_sample(self, sample: SampleWithSHAP) -> None:
        super().set_sample(sample)
        self.flat_base = self._flatten(self._full_base_vals())
        self.xi = self.rng.normal(size=(self.prec.incidence.shape[0],
                                        self.num_draws))

    def __call__(self, pinned):
        all_values = []
        all_context = []
        for n in range(len(pinned)):
            result = condition_and_sample(
                self.prec,
                self._flatten(self._full_mask(pinned[n])),
                self.flat_base,
                self.xi,
            )
            filled_samples = self._unflatten(result.values)
            all_values.append(filled_samples[:, self.c:])
            if self.c:
                all_context.append(filled_samples[:, :self.c])

        all_values = np.stack(all_values, 0)

        if len(all_context) > 0:
            all_context = np.stack(all_context, 0)
        else:
            all_context = None

        return FillResult(all_values, all_context)

    def _full_base_vals(self) -> np.ndarray:

        full = np.concatenate((self._context_vals(), self.base_vals), axis=0)
        return full  # [T+C, F]

    def _full_mask(self, pinned: np.ndarray) -> np.ndarray:

        full = np.concatenate((self._context_pinned(), pinned), axis=0)
        return full  # [F, T+C]

    def _flatten(self, x: np.ndarray) -> np.ndarray:
        # x [F, C+T]
        return x.T.reshape(-1)

    def _unflatten(self, x: np.ndarray) -> np.ndarray:
        return x.reshape(self.base_vals.shape[1],
                         self.base_vals.shape[0] + self.c,
                         -1).transpose(2, 1, 0)  # [B, C+T, F]

    @abstractmethod
    def _context_vals(self) -> np.ndarray:
        pass

    @abstractmethod
    def _context_pinned(self) -> np.ndarray:
        pass


class ObservationalVAR(Observational):

    def _context_vals(self):
        """These values dont matter, context is always unpinned"""
        return np.zeros((self.c, self.base_vals.shape[1]))

    def _context_pinned(self):
        """All context unpinned (sampled over)"""
        return np.zeros((self.c, self.base_vals.shape[1]), dtype=bool)


class ObservationalGT(Observational):

    def __init__(self, sample: SampleWithSHAP, feature_names, num_draws, prec,
                 n_context, graph):
        super().__init__(sample, feature_names, num_draws, prec, n_context)
        self.graph = graph
        self.base_context = sample.ctx_vals(feature_names, n_context)

    def _context_vals(self):
        return self.base_context

    def _context_pinned(self):
        mask = np.zeros_like(self.base_context, bool)
        mask[:self.graph.max_lag] = True
        return mask
