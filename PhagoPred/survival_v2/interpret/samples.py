"""Choosing the (cell, landmark) pairs to explain, and caching them on disk.

The same code serves synthetic and real data: noisy features are always
stored; clean features and pre-frame-0 context only when the dataset has them
(``dataset.has_clean``), which is what the ground truth needs.
"""
from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np

from PhagoPred.utils.logger import get_logger
from ..data.dataset import CellDataset
from .config import SampleSelectionCfg
from .data_models import SampleValues, SampleWithSHAP
from .utils.io import (SHAPFileMetadata, read_metadata, read_sample,
                       sample_indices, write_metadata, write_sample)

LOG = get_logger(__name__)


def build_samples(dataset: CellDataset,
                  cfg: SampleSelectionCfg,
                  seed: int = 0) -> list[SampleWithSHAP]:
    """Up to ``cfg.num_samples`` samples, at most one per cell.

    Cells are visited in a random order weighted by their number of valid
    landmarks (as the training sampler does), and each gets a random landmark
    ``L`` whose window ``[0, L)`` is fully observed. The window starts at frame
    0 because the ground truth rolls the graph forward from the stored
    context, which ends at frame 0.
    """
    rng = np.random.default_rng(seed)
    meta = dataset.cell_metadata
    with_truth = dataset.has_clean

    weights = np.array([len(lf) for lf in meta.landmark_frames], float)
    order = rng.choice(len(dataset),
                       size=int((weights > 0).sum()),
                       replace=False,
                       p=weights / weights.sum())

    samples = []
    for idx in order:
        if len(samples) == cfg.num_samples:
            break
        landmarks = _eligible_landmarks(dataset, idx, cfg)
        if len(landmarks) == 0:
            continue
        L = int(rng.choice(landmarks))
        samples.append(_make_sample(dataset, idx, L, with_truth))

    if len(samples) < cfg.num_samples:
        LOG.warning(f'only {len(samples)}/{cfg.num_samples} cells have an '
                    'eligible landmark')
    return samples


def load_or_build_samples(path: Path | str,
                          dataset: CellDataset,
                          meta: SHAPFileMetadata,
                          seed: int = 0) -> list[SampleWithSHAP]:
    """The first ``num_samples`` samples saved at ``path`` (with any SHAP
    values already computed for them), if that file was built with the same
    selection and prediction target; otherwise build them and start a new
    file there."""
    path = Path(path)
    selection = meta.interpret.selection
    if path.exists():
        with h5py.File(path, 'r') as f:
            saved = read_metadata(f)
            if _same_selection(saved, meta) and \
                    saved.num_samples >= selection.num_samples:
                idxs = sample_indices(f)[:selection.num_samples]
                LOG.info(f'loaded {len(idxs)} samples from {path}')
                return [read_sample(f, i) for i in idxs]
        LOG.info(f'{path} was built with a different selection; rebuilding')

    samples = build_samples(dataset, selection, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        write_metadata(f, meta)
        for i, sample in enumerate(samples):
            write_sample(f, sample, i)
    LOG.info(f'built {len(samples)} samples, saved to {path}')
    return samples


def _same_selection(saved: SHAPFileMetadata,
                    requested: SHAPFileMetadata) -> bool:
    """Samples depend only on the selection cfg and the prediction target
    (which sets the valid landmarks), not on SHAP or background settings.
    ``num_samples`` is compare=False, so more saved samples still match."""
    return (saved.interpret.selection == requested.interpret.selection and
            saved.prediction_target == requested.prediction_target and
            saved.scenario == requested.scenario)


def _eligible_landmarks(dataset: CellDataset, idx: int,
                        cfg: SampleSelectionCfg) -> np.ndarray:
    meta = dataset.cell_metadata
    if int(meta.start_frames[idx]) != 0:
        return np.array([], int)

    landmarks = dataset.valid_landmarks(idx)
    if cfg.min_sequence_len is not None:
        landmarks = landmarks[landmarks >= cfg.min_sequence_len]

    # Fully observed window: L <= length of the NaN-free prefix.
    missing = np.isnan(dataset.raw_features(idx)).any(axis=1)
    prefix = int(missing.argmax()) if missing.any() else len(missing)
    landmarks = landmarks[landmarks <= prefix]

    if cfg.min_horizon_cif > 0.0 and len(landmarks):
        probs = _horizon_event_probability(dataset, idx, landmarks)
        if probs is not None:
            landmarks = landmarks[probs >= cfg.min_horizon_cif]
    return landmarks


def _horizon_event_probability(dataset: CellDataset, idx: int,
                               landmarks: np.ndarray) -> np.ndarray | None:
    """True P(death in (L, L + horizon] | alive at L), or None on real data."""
    pmf = dataset._cell_pmf(idx)  # pylint: disable=protected-access
    if pmf is None:
        return None
    cif = np.cumsum(pmf, dtype=np.float64)
    end = np.minimum(landmarks + dataset.horizon, len(cif) - 1)
    survival = 1.0 - cif[landmarks]
    return np.divide(cif[end] - cif[landmarks],
                     survival,
                     out=np.zeros(len(landmarks)),
                     where=survival > 0.0)


def _make_sample(dataset: CellDataset, idx: int, L: int,
                 with_truth: bool) -> SampleWithSHAP:
    death = float(dataset.cell_metadata.death_frames[idx])
    values = SampleValues(features=dataset.raw_features(idx)[:L].T)  # [F, L]
    context_names = None
    if with_truth:
        values.clean_features = dataset.raw_features(idx, clean=True)[:L].T
        values.context_features = dataset.raw_context(idx).T  # [F + 1, C]
        context_names = dataset.context_feature_names
    return SampleWithSHAP(
        feature_names=list(dataset.feature_names),
        feature_vals=values,
        landmark_frame=L,
        event_frame=None if np.isnan(death) else int(death),
        context_feature_names=context_names,
    )
