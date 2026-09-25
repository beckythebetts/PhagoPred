from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import h5py
import numpy as np
import torch
import shap

from PhagoPred.survival_v2.models import SurvivalModel
from PhagoPred.survival_v2.interpret import (SampleWithSHAP, SHAPResult,
                                             ExplainerEnum, BackgroundEnum)
from .var_precision import VARFit, VARPrecision, build_precision, sample_conditional
from .utils import align_sample_features


class BackgroundStrategy(ABC):
    """Fills in a coalition's absent entries and returns the model's prediction.

    ``v(S)`` is itself an expectation over the background distribution, so
    implementations may average several filled variants (e.g. several
    unconditional background draws, or several conditional samples) rather
    than predicting on a single filled input.
    """
    background: BackgroundEnum

    @abstractmethod
    def predict(self, predict_fn, base_input: torch.Tensor,
                keep: torch.Tensor) -> torch.Tensor:
        """``predict_fn(x: (N,T,F)) -> (N, out_dim)``; ``keep``: (N,T,F) mask."""

    def new_explanation(self, T: int) -> None:
        """Called once per Shapley explanation (one ``compute_importance``
        call), before any coalitions are queried. Strategies whose ``v(S)``
        involves its own randomness (``ObservationalBackground``) use this to
        fix common random numbers shared by every coalition queried during
        that explanation; a no-op otherwise (``InterventionalBackground``'s
        ``v(S)`` is already deterministic given ``keep``)."""


class InterventionalBackground(BackgroundStrategy):
    """Absent entries -> a constant, or averaged over independent background
    draws (unconditional on what's pinned — the do-operator/marginal
    definition of v(S))."""

    background = BackgroundEnum.INTERVENTIONAL

    def __init__(self,
                 mask_value: float = 0.0,
                 value_background: Optional[torch.Tensor] = None):
        self.mask_value = mask_value
        self.value_background = value_background  # (B, T, F) or None

    def predict(self, predict_fn, base_input, keep):
        if self.value_background is None:
            return predict_fn(base_input * keep + self.mask_value * (1 - keep))
        acc = None
        for bg in self.value_background:  # (T, F)
            p = predict_fn(base_input * keep + bg[None] * (1 - keep))
            acc = p if acc is None else acc + p
        return acc / self.value_background.shape[0]


class ObservationalBackground(BackgroundStrategy):
    """Absent entries -> sampled from the conditional distribution given the
    pinned entries, via a VAR-estimated precision — the counterpart of
    ``ground_truth._coalition_value`` for real data, evaluated through this
    model's forward pass rather than ``graph.rules``.

    ``var_fit`` is the *one global* VAR fit (see ``var_precision.fit_var``),
    shared across every sample explained with this strategy. Building the
    actual sparse precision (``build_precision``) is cheap and re-derived per
    sample-length ``T`` on first use (cached thereafter) since different real
    samples generally have different observed lengths — only the expensive
    global coefficient fit is meant to happen once.
    """

    background = BackgroundEnum.OBSERVATIONAL

    def __init__(self, var_fit: VARFit, num_cond_samples: int = 8):
        self.var_fit = var_fit
        self.num_cond_samples = num_cond_samples
        self._cache: dict[int, VARPrecision] = {}
        self._xi: np.ndarray | None = None

    def _precision_for(self, T: int) -> VARPrecision:
        if T not in self._cache:
            self._cache[T] = build_precision(self.var_fit, T)
        return self._cache[T]

    def new_explanation(self, T: int) -> None:
        """Redraw the shared conditional-sampling noise for one explanation.

        Without this, every coalition ``predict`` evaluates draws its own
        independent ``xi`` (see ``sample_conditional``), so a marginal
        contribution like ``v(S union {i}) - v(S)`` is a difference of two
        independently-noisy estimates that never cancels — even a player the
        VAR precision treats as fully independent of everything else picks up
        a nonzero noise floor. Fixing one ``xi`` for every coalition queried
        during this explanation (common random numbers) makes that
        cancellation exact instead, matching
        ``ground_truth.generate_samples``'s per-permutation CRN.
        """
        prec = self._precision_for(T)
        n_eq = prec.prec.incidence.shape[0]
        self._xi = np.random.normal(size=(n_eq, self.num_cond_samples))

    def predict(self, predict_fn, base_input, keep):
        # base_input: (1, T, K); keep: 1=pinned, 0=absent, matches
        # sample_conditional's pinned_mask convention directly once broadcast
        # to (N, T, K) — callers (e.g. FeatureMaskWrapper/TemporalMaskWrapper)
        # may pass a smaller broadcastable shape like (N, 1, K) or (N, T, 1).
        device, dtype = base_input.device, base_input.dtype
        T, K = base_input.shape[1], base_input.shape[2]
        prec = self._precision_for(T)
        B = self.num_cond_samples

        if self._xi is None or self._xi.shape != (prec.prec.incidence.shape[0],
                                                  B):
            self.new_explanation(T)

        base_np = base_input[0].detach().cpu().numpy()  # (T, K)
        basevec = base_np.T.reshape(-1)  # feature-major: col[(f,t)] = fi*T+t

        keep_np = keep.detach().cpu().numpy()
        N = keep_np.shape[0]
        keep_np = np.broadcast_to(keep_np, (N, T, K)).astype(bool)

        # One conditional draw per (coalition, background sample); the model
        # forward pass is batched over all N*B at once, only the sparse solve
        # (a different Q_FF per coalition) has to loop over coalitions. Every
        # coalition reuses self._xi (common random numbers, see
        # new_explanation) rather than drawing its own.
        filled = np.empty((N, B, T, K), dtype=np.float32)
        for i in range(N):
            pinned_mask = keep_np[i].T.reshape(-1)
            samples = sample_conditional(prec,
                                         pinned_mask,
                                         basevec,
                                         B,
                                         xi=self._xi)  # (K,T,B)
            filled[i] = samples.transpose(2, 1, 0)  # (B, T, K)

        flat = torch.as_tensor(filled.reshape(N * B, T, K),
                               device=device,
                               dtype=dtype)
        preds = predict_fn(flat)  # (N*B, out_dim)
        return preds.reshape(N, B, -1).mean(dim=1)


class MaskWrapperBase:
    """Base class for mask wrappers used in Kernel SHAP analysis.

    Allows models to be called with different masks applied; how a masked
    (absent) entry is filled in is delegated to ``background``.
    """

    def __init__(
        self,
        model: SurvivalModel,
        base_input: torch.Tensor,
        lengths: torch.Tensor,
        background: BackgroundStrategy,
        output_type: Literal["expected_time", "cif", "pmf"] = "expected_time",
        target_bin: Optional[int] = None,
        device: str = "cpu",
        time_bins: Optional[np.ndarray] = None,
        max_batch: int = 2048,
    ):
        self.model = model
        self.model.eval()
        self.base_input = base_input.to(device)
        self.lengths = lengths.to(device)
        self.device = device
        self.background = background
        self.output_type = output_type
        self.target_bin = target_bin
        self.time_bins = time_bins
        self.max_batch = max_batch

    def __call__(self, masks: np.ndarray) -> np.ndarray:
        """Apply a batch of coalition masks and return model predictions.

        ``masks`` has shape (N, num_players). All N coalitions are evaluated in
        batched forward passes (chunked by ``max_batch``) rather than one at a
        time, which is what actually keeps the GPU busy.
        """
        masks = np.asarray(masks)
        if masks.ndim == 1:
            masks = masks[None, :]
        n = masks.shape[0]

        # Some background strategies (e.g. ObservationalBackground) expand each
        # coalition into `num_cond_samples` draws and score them in a single
        # predict_fn call, so the real forward-pass batch is chunk_size * that
        # multiplier, not chunk_size. Shrink the chunk accordingly so max_batch
        # still bounds what actually hits the GPU.
        cond_samples = getattr(self.background, "num_cond_samples", 1)
        chunk_size = max(1, self.max_batch // cond_samples)

        preds = []
        for start in range(0, n, chunk_size):
            chunk = masks[start:start + chunk_size]
            # (B, T, F) keep-mask: 1 -> original value, 0 -> masked
            keep = torch.as_tensor(self._expand_mask(chunk),
                                   device=self.device,
                                   dtype=self.base_input.dtype)
            v = self.background.predict(self._predict, self.base_input, keep)
            preds.append(v.detach().cpu().numpy())

        return np.concatenate(preds, axis=0).reshape(n, -1)[:, :1]

    def _predict(self, masked_input: torch.Tensor) -> torch.Tensor:
        lengths = self.lengths[:1].expand(masked_input.shape[0])
        with torch.no_grad():
            pred = self.model(masked_input, lengths, return_attention=False)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = self._extract_output(pred)
        return pred.reshape(pred.shape[0], -1)

    @abstractmethod
    def _expand_mask(self, masks: np.ndarray) -> np.ndarray:
        """Expand coalition masks (N, num_players) to a (N, T, F) keep-mask.

        Entries are 1 where the original input is kept and 0 where it is
        absent (filled in by ``self.background``). A trailing dim of size 1
        may be returned (e.g. (N, T, 1)) to broadcast across features/frames.
        """

    def _extract_output(self, logits: torch.Tensor) -> torch.Tensor:
        """Extract the appropriate scalar output from model logits."""
        if self.output_type == "expected_time":
            if self.time_bins is not None:
                return self.model.predict_restricted_survival_time(
                    logits, self.time_bins)
            return self.model.predict_expected_time(logits)
        elif self.output_type == "binary":
            return self.model.predict_binary(logits)
        elif self.output_type == "cif":
            cif = self.model.predict_cif(logits)
            if self.target_bin is not None:
                return cif[:, self.target_bin]
            return cif[:, -1]  # Final CIF value
        elif self.output_type == "pmf":
            pmf = self.model.predict_pmf(logits)
            if self.target_bin is not None:
                return pmf[:, self.target_bin]
            return pmf.max(dim=1).values  # Max probability
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")


class TemporalMaskWrapper(MaskWrapperBase):
    """Importance of temporal segments: a 1D mask of length ``num_segments``
    masks all features at the corresponding timesteps."""

    def __init__(self, *args, num_segments: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.T = self.base_input.shape[1]
        self.segment_boundaries = np.linspace(0,
                                              self.T,
                                              num_segments + 1,
                                              dtype=int)

    def _expand_mask(self, masks: np.ndarray) -> np.ndarray:
        # masks: (N, num_segments) -> (N, T, 1), broadcast across features
        n = masks.shape[0]
        full = np.zeros((n, self.T), dtype=np.float32)
        for i in range(self.num_segments):
            start = self.segment_boundaries[i]
            end = self.segment_boundaries[i + 1]
            full[:, start:end] = masks[:, i:i + 1]
        return full[:, :, None]


class FeatureMaskWrapper(MaskWrapperBase):
    """Importance of each feature: a 1D mask of length ``num_features`` masks
    that feature across all timesteps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = self.base_input.shape[2]

    def _expand_mask(self, masks: np.ndarray) -> np.ndarray:
        return masks.astype(np.float32)[:, None, :]


class TemporalFeatureMaskWrapper(MaskWrapperBase):
    """Joint (segment, feature) importance map."""

    def __init__(self, *args, num_segments: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.T = self.base_input.shape[1]
        self.F = self.base_input.shape[2]
        self.segment_boundaries = np.linspace(0,
                                              self.T,
                                              num_segments + 1,
                                              dtype=int)

    def _expand_mask(self, masks: np.ndarray) -> np.ndarray:
        n = masks.shape[0]
        m = masks.reshape(n, self.F, self.num_segments).astype(
            np.float32)  # (n, F, segments)
        full = np.zeros((n, self.T, self.F), dtype=np.float32)
        for i in range(self.num_segments):
            start = self.segment_boundaries[i]
            end = self.segment_boundaries[i + 1]
            full[:, start:end, :] = m[:, :, i][:, None, :]
        return full


_WRAPPERS = {
    'temporal': TemporalMaskWrapper,
    'feature': FeatureMaskWrapper,
    'temporal_feature': TemporalFeatureMaskWrapper,
}


class KernelSHAP:
    """Perturbation-based SHAP analysis for survival models.

    Uses ``shap.KernelExplainer`` to compute SHAP values by masking temporal
    segments and/or features; how a mask's absent entries get filled in
    (interventional vs. observational) is supplied per call via ``background``,
    not fixed on the explainer — the KernelExplainer machinery itself doesn't
    care which was used.
    """

    def __init__(
        self,
        model: SurvivalModel,
        feature_names: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.feature_names = feature_names

    def compute_importance(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        background: BackgroundStrategy,
        num_segments: Optional[int] = 50,
        nsamples: int = 500,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        show_progress: bool = True,
        importance_type: Literal["temporal", "feature",
                                 "temporal_feature"] = "temporal",
        time_bins: Optional[np.ndarray] = None,
        l1_reg: bool | str | int | float = False,
    ) -> tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Compute SHAP values for one axis. Returns (shap_values, baseline,
        segment_boundaries)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor([lengths])
        elif lengths.dim() == 0:
            lengths = lengths.unsqueeze(0)

        kwargs = dict(
            model=self.model,
            base_input=x,
            lengths=lengths,
            background=background,
            output_type=output_type,
            target_bin=target_bin,
            device=self.device,
            time_bins=time_bins,
        )

        if importance_type in (
                "temporal", "temporal_feature") and num_segments is not None:
            # Clamp to the sample's own length, matching
            # ground_truth.generate_samples._axis_segments's min(num_segments,
            # lf) — otherwise np.linspace(0, T, num_segments+1) below produces
            # duplicate (zero-width) boundaries for a short T < num_segments,
            # silently turning those segments into structural dummy players
            # (no frames assigned, so masking them can never do anything) and
            # dividing by zero when plots.spread_segments later expands them.
            num_segments = min(num_segments, x.shape[1])

        if importance_type == "feature":
            player_background = np.zeros((1, x.shape[2]))
            wrapper = FeatureMaskWrapper(**kwargs)
        elif importance_type == "temporal":
            player_background = np.zeros((1, num_segments))
            wrapper = TemporalMaskWrapper(**kwargs, num_segments=num_segments)
        else:
            player_background = np.zeros((1, num_segments * x.shape[2]))
            wrapper = TemporalFeatureMaskWrapper(**kwargs,
                                                 num_segments=num_segments)

        # Fresh common-random-numbers for this axis's explanation, so every
        # coalition KernelExplainer queries shares the same background
        # randomness (see BackgroundStrategy.new_explanation) — including the
        # null-coalition baseline, which KernelExplainer.__init__ queries
        # immediately below to set self.fnull/expected_value, so this must
        # happen *before* that construction or the baseline silently reuses
        # whatever xi was left over from the previous axis/sample.
        background.new_explanation(x.shape[1])

        explainer = shap.KernelExplainer(wrapper, player_background)
        test_input = np.ones_like(player_background)

        # l1_reg=False -> full weighted-least-squares Shapley values (dense).
        # The shap default ("auto") applies an L1/LARS penalty that zeros out
        # players when nsamples is small relative to 2**num_players.
        shap_values = explainer.shap_values(test_input,
                                            nsamples=nsamples,
                                            l1_reg=l1_reg,
                                            silent=not show_progress)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        baseline = explainer.expected_value
        if isinstance(baseline, np.ndarray):
            baseline = baseline[0]

        boundaries = getattr(wrapper, 'segment_boundaries', None)
        return shap_values.squeeze(), baseline, boundaries

    def analyse(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        background: BackgroundStrategy,
        num_segments: int = 50,
        nsamples_temporal: Optional[int] = None,
        nsamples_feature: Optional[int] = None,
        nsamples_temporal_feature: Optional[int] = None,
        output_type: str = "expected_time",
        target_bin: Optional[int] = None,
        time_bins: Optional[np.ndarray] = None,
        l1_reg: bool | str | int | float = False,
        show_progress: bool = True,
    ) -> SHAPResult:
        """Full temporal/feature/temporal_feature SHAP analysis for one sample."""
        num_features = x.shape[-1]
        # 2*M + 2048 is shap.KernelExplainer's own "auto" nsamples heuristic
        # for a game with M players; applied per-axis here (M = num_segments,
        # num_features, or their product) rather than one flat count shared
        # across axes of very different sizes.
        if nsamples_temporal is None:
            nsamples_temporal = 2 * num_segments + 2048
        if nsamples_feature is None:
            nsamples_feature = 2 * num_features + 2048
        if nsamples_temporal_feature is None:
            nsamples_temporal_feature = 2 * num_segments * num_features + 2048

        temporal, _, boundaries = self.compute_importance(
            x, lengths, background, num_segments, nsamples_temporal,
            output_type, target_bin, show_progress, 'temporal', time_bins,
            l1_reg)
        feature, _, _ = self.compute_importance(x, lengths, background,
                                                num_segments, nsamples_feature,
                                                output_type, target_bin,
                                                show_progress, 'feature',
                                                time_bins, l1_reg)
        temporal_feature, _, _ = self.compute_importance(
            x, lengths, background, num_segments, nsamples_temporal_feature,
            output_type, target_bin, show_progress, 'temporal_feature',
            time_bins, l1_reg)
        temporal_feature = temporal_feature.reshape(num_features, num_segments)

        return SHAPResult(
            explainer=ExplainerEnum.KERNEL,
            background=background.background,
            temporal=temporal,
            feature=feature,
            temporal_feature=temporal_feature,
            segment_boundaries=boundaries,
        )


def analyse_sample_in_file(
    h5_path: Path,
    sample_idx: int,
    kernel_shap: KernelSHAP,
    background: BackgroundStrategy,
    means: np.ndarray,
    stds: np.ndarray,
    model_feat_names: list[str] | None = None,
    device: str = 'cpu',
    **analyse_kwargs,
) -> None:
    """Read one already-written sample, run KernelSHAP with ``background``, and
    merge the result back in. Phase 2 of the write-samples-first / analyse-and-
    merge-after pattern: ``model.utils._generate_samples`` writes the sample
    skeletons (and, taken together, the background pool) before any explainer
    runs; this fills one of them in.

    ``model_feat_names`` aligns the sample to the model's own input columns
    and crops to the observed past before explaining it (see
    ``align_sample_features``) — pass it whenever the results file may hold
    extra/differently-ordered/future-spanning columns, e.g. a ground-truth
    file (whose ``feature_names`` include ``Hazard``); omit it for a real-data
    file already in that exact shape.
    """
    sample = SampleWithSHAP.read_h5(h5_path, sample_idx)
    feature_vals = (align_sample_features(sample, model_feat_names)
                    if model_feat_names is not None else sample.feature_vals)
    # (F, T) -> (T, F), the model's expected layout, normalised as in training
    feature_vals = (feature_vals.T - means) / stds
    x = torch.tensor(feature_vals, dtype=torch.float32, device=device)
    length = torch.tensor([x.shape[0]], device=device)

    result = kernel_shap.analyse(x, length, background, **analyse_kwargs)
    sample.shap_vals[(ExplainerEnum.KERNEL, background.background)] = result
    sample.write_h5(h5_path, sample_idx)
    sample.write_h5(h5_path, sample_idx)
