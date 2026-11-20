from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from PhagoPred.survival_v2.data.graph_synthetic.scenarios import ScenarioCfg
from PhagoPred.survival_v2.data.graph_synthetic.dynamics import linearised_precision
from ...utils.precision import ObservedPrecision, node_noise, sample_free
from ..var import VARFit, build_precision, default_burn_in
from ..data_models import SampleWithSHAP
from ..utils.seeds import get_generator, FILL
from ..backgrounds import BackgroundSet
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

    def __init__(self, feature_names: list[str], num_draws: int, clean: bool,
                 **kwargs):
        self.feature_names = feature_names
        self.num_draws = num_draws
        self.clean = clean

        self.base_vals = None
        self.rng = None

    def set_sample(self, sample: SampleWithSHAP) -> None:
        self.base_vals = sample.feat_vals(self.feature_names, clean=self.clean)
        self.rng = get_generator(sample, FILL)

    @abstractmethod
    def __call__(self, pinned: np.ndarray) -> FillResult:
        pass


class Interventional(BGFill):

    def __init__(self, feature_names, num_draws, bg_set: BackgroundSet):
        super().__init__(feature_names, num_draws, clean=bg_set.clean)
        self.bg_set = bg_set
        self.current_bg_samples = None

    def set_sample(self, sample: SampleWithSHAP) -> None:
        super().set_sample(sample)
        eligible = self.bg_set.eligible(sample.landmark_frame)
        if len(eligible) < self.num_draws:
            raise ValueError(
                f'only {len(eligible)} background rows cover '
                f'{sample.landmark_frame} frames; need {self.num_draws}')
        idxs = self.rng.choice(eligible, self.num_draws, replace=False)
        self.current_bg_samples = self.bg_set.rows(idxs,
                                                   sample.landmark_frame,
                                                   self.rng)

    def __call__(self, pinned):
        values = np.where(pinned[:, None], self.base_vals[None, None],
                          self.current_bg_samples[None])

        return FillResult(values)


class Observational(BGFill, ABC):
    """Fills the unpinned nodes with draws from a Gaussian over the window's
    (feature, frame) nodes, conditioned on the pinned ones. The precision has
    one node per (feature, frame) of context + window, so it is built per
    sample (``_precision``), from a dependence model shared across samples."""

    def __init__(self, feature_names, num_draws, n_context: int, clean: bool,
                 **kwargs):
        super().__init__(feature_names, num_draws, clean)
        self.c = n_context

        self.prec: ObservedPrecision | None = None  # for the current sample
        self.full_base = None  # [C+T, F] base values, context included
        self.flat_base = None  # the same, flattened feature-major
        self.xi = None  # Gaussian draw for sampling from self.prec
        # Per sample, reused by every coalition:
        self._Qcsr = None  # prec.Q as CSR
        self._noise = None  # node_noise(prec, xi), (n_obs, B)

    def set_sample(self, sample: SampleWithSHAP) -> None:
        super().set_sample(sample)
        self.full_base = self._full_base_vals()
        self.prec = self._precision(self.full_base)
        self.flat_base = self._flatten(self.full_base)
        self.xi = self.rng.normal(size=(self.prec.incidence.shape[0],
                                        self.num_draws))
        self._Qcsr = self.prec.Q.tocsr()
        self._noise = node_noise(self.prec, self.xi)

    def __call__(self, pinned):
        N, B = len(pinned), self.num_draws
        CT = len(self.full_base)
        out = np.empty((N, B, CT, self.full_base.shape[1]))  # [N, B, C+T, F]
        out[:] = self.full_base
        for n in range(N):
            mask = self._flatten(self._full_mask(pinned[n]))
            free = np.flatnonzero(~mask)
            if len(free):
                draws = sample_free(self.prec, self._Qcsr, free,
                                    np.flatnonzero(mask), self.flat_base,
                                    self._noise)[0]  # (n_free, B)
                # node f * CT + t -> (frame t, feature f)
                out[n][:, free % CT, free // CT] = draws.T
        values = out[:, :, self.c:]
        context = out[:, :, :self.c] if self.c else None
        return FillResult(values, context)

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
    def _precision(self, full_base: np.ndarray) -> ObservedPrecision:
        """Precision over the nodes of ``full_base`` [C+T, F], feature-major
        (node ``f * (C+T) + t``)."""

    @abstractmethod
    def _context_vals(self) -> np.ndarray:
        pass

    @abstractmethod
    def _context_pinned(self) -> np.ndarray:
        pass


class ObservationalVAR(Observational):
    """Dependence from a VAR fitted to the background set: noisy rows for
    the model, clean rows for the ground-truth oracle. The context is free
    burn-in, so the window's first frames start near stationary variance."""

    def __init__(self,
                 feature_names,
                 num_draws,
                 var_fit: VARFit,
                 clean: bool,
                 n_context: int | None = None):
        if list(var_fit.feature_names) != list(feature_names):
            raise ValueError(f'VAR fitted on {var_fit.feature_names}, fill '
                             f'uses {feature_names}')
        if n_context is None:
            n_context = default_burn_in(var_fit)
        super().__init__(feature_names, num_draws, n_context, clean)
        self.var_fit = var_fit
        self._cache: dict[int, ObservedPrecision] = {}

    def _precision(self, full_base):
        # Depends only on the window length: cache it.
        n = len(full_base)
        if n not in self._cache:
            self._cache[n] = build_precision(self.var_fit, n)
        return self._cache[n]

    def _context_vals(self):
        """These values dont matter, context is always unpinned"""
        return np.zeros((self.c, self.base_vals.shape[1]))

    def _context_pinned(self):
        """All context unpinned (sampled over)"""
        return np.zeros((self.c, self.base_vals.shape[1]), dtype=bool)


class ObservationalGT(Observational):
    """Dependence from the true graph, linearised at the sample's own clean
    trajectory (exact for linear scenarios). The oldest ``max_lag`` context
    frames are pinned to the true past; the rest of the context is free.

    Nonlinear scenarios: the linearisation is only a proposal, and importance
    weights (FillResult.log_weights) are not implemented yet."""

    def __init__(self, feature_names, num_draws, n_context,
                 scenario: ScenarioCfg):
        super().__init__(feature_names, num_draws, n_context, clean=True)
        self.graph = scenario.graph
        self.max_lag = scenario.graph.max_lag
        if n_context < self.max_lag:
            raise ValueError(f'n_context ({n_context}) < max_lag '
                             f'({self.max_lag}): nothing left to pin')
        self.base_context = None

    def set_sample(self, sample):
        self.base_context = sample.ctx_vals(self.feature_names, self.c)
        super().set_sample(sample)

    def _precision(self, full_base):
        signals = {n: full_base[:, i] for i, n in enumerate(self.feature_names)}
        return linearised_precision(self.graph, self.feature_names, signals)

    def _context_vals(self):
        return self.base_context

    def _context_pinned(self):
        mask = np.zeros_like(self.base_context, bool)
        mask[:self.max_lag] = True
        return mask
