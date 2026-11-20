"""Explain a trained model, and (synthetic data) the ground truth, on the
held-out SHAP shard.

Samples and ground-truth SHAP don't depend on the model, so they live in one
shared file per (shard, prediction target), next to the shard:

    <shard dir>/shap_samples/<shard>_<target>.h5   samples + GT SHAP

and are reused by every model trained on that dataset. Model SHAP goes in

    <experiment_dir>/shap.h5                        samples + GT + model SHAP

which is self-contained for plotting.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

import h5py

from PhagoPred.utils.logger import get_logger
from ..configs import ExperimentCfg, PredictionTarget
from ..configs.datasets import BinaryDatasetCfg, DatasetCfg, SurvivalDatasetCfg
from ..data import BinaryCellDataset, CellDataset, SurvivalCellDataset
from ..data.graph_synthetic.scenarios import ScenarioCfg, load_scenario
from ..utils.io import load_model
from .backgrounds import BackgroundSet
from .config import INTERPRETS, InterpretCfg, ShapCfg
from .data_models import BackgroundEnum, PredictorEnum, SampleWithSHAP
from .explain_sample import ExplainCfg, explain_sample
from .samples import _same_selection, load_or_build_samples
from .utils.io import (SHAPFileMetadata, read_metadata, read_sample,
                       sample_indices, write_metadata, write_sample)
from .value_func import ValueFunction
from .value_func.bg_fill import (Interventional, ObservationalGT,
                                 ObservationalVAR)
from .value_func.predictor import GraphPredictor, ModelPredictor
from .var import default_burn_in, fit_var

LOG = get_logger(__name__)

SHAP_FILE = 'shap.h5'


def run_shap(experiment_dir: Path | str,
             cfg: InterpretCfg | None = None,
             device: str | None = None) -> list[SampleWithSHAP]:
    """GT SHAP (synthetic only, cached across models), then model SHAP, for
    every background in ``cfg.shap.backgrounds``. Results already computed
    with the same settings are reused, so an interrupted run resumes."""
    cfg = cfg or INTERPRETS['Standard']
    experiment_dir = Path(experiment_dir)
    model, exp_cfg = load_model(experiment_dir, device)
    dcfg = exp_cfg.dataset
    target = dcfg.prediction_target
    players = list(exp_cfg.feature_combo)
    if not dcfg.shap_paths:
        raise ValueError(f'{experiment_dir}: dataset cfg has no shap_paths')

    shap_path = Path(dcfg.shap_paths[0])
    scenario = load_scenario(shap_path)  # None for real data
    meta = SHAPFileMetadata(cfg, target,
                            None if scenario is None else scenario.filename)

    shap_ds = _raw_dataset(dcfg, dcfg.shap_paths, players)
    samples = load_or_build_samples(_shared_path(shap_path, target), shap_ds,
                                    meta, cfg.shap.seed)

    train_ds = _raw_dataset(dcfg, dcfg.train_paths, players)
    # With hazard_from_frame_0 survival depends on time since frame 0 (the
    # hazard window fills up over the first max_lag frames), so a background
    # window covers the same frames [0, L) as the sample.
    align = ('start' if scenario is not None
             and scenario.graph.hazard_from_frame_0 else 'landmark')
    backgrounds = _Backgrounds(train_ds, cfg, players, align)
    explain_cfg = ExplainCfg(cfg.shap.num_temporal_segments,
                             dict(cfg.shap.budgets))

    if scenario is not None:
        gt_vfs = _gt_value_functions(scenario, target, players, cfg.shap,
                                     backgrounds)
        _explain_all(_shared_path(shap_path, target), samples, meta, gt_vfs,
                     PredictorEnum.GROUND_TRUTH, players, explain_cfg)

    model_vfs = _model_value_functions(model, exp_cfg, players, cfg.shap,
                                       backgrounds)
    _explain_all(experiment_dir / SHAP_FILE, samples, meta, model_vfs,
                 PredictorEnum.MODEL, players, explain_cfg)
    return samples


# === value functions ===


class _Backgrounds:
    """Background sets and VAR fits, built on first use. The clean (GT) and
    noisy (model) sets use the same seed, so they hold the same rows."""

    def __init__(self, dataset: CellDataset, cfg: InterpretCfg,
                 players: list[str], align: str = 'landmark'):
        self.dataset, self.cfg, self.players = dataset, cfg, players
        self.align = align
        self._sets, self._vars = {}, {}

    def set(self, clean: bool) -> BackgroundSet:
        if clean not in self._sets:
            self._sets[clean] = BackgroundSet.from_dataset(
                self.dataset, self.cfg.background.pool_size, clean,
                self.cfg.shap.seed, self.align)
        return self._sets[clean]

    def var(self, clean: bool):
        if clean not in self._vars:
            bg = self.cfg.background
            self._vars[clean] = fit_var(self.set(clean),
                                        self.players,
                                        bg.var_max_lag,
                                        criterion=bg.var_criterion)
            LOG.info(f'VAR (clean={clean}): p={self._vars[clean].p} '
                     f'({bg.var_criterion}, max {bg.var_max_lag})')
        return self._vars[clean]


def _fill(bg: BackgroundEnum, players: list[str], shap: ShapCfg,
          backgrounds: _Backgrounds, clean: bool,
          scenario: ScenarioCfg | None):
    B = shap.num_draws
    if bg == BackgroundEnum.INTERVENTIONAL:
        return Interventional(players, B, backgrounds.set(clean))
    if bg == BackgroundEnum.OBSERVATIONAL_VAR:
        var = backgrounds.var(clean)
        n_context = (backgrounds.cfg.background.var_burn_in or
                     default_burn_in(var))
        if scenario is not None:
            # GraphPredictor reads max_lag frames of context before frame 0.
            n_context = max(n_context, scenario.graph.max_lag)
        return ObservationalVAR(players,
                                B,
                                var,
                                clean=clean,
                                n_context=n_context)
    if bg == BackgroundEnum.OBSERVATIONAL_GT:
        # All the clean context stored before frame 0 (see generate_datasets).
        n_context = min(2 * scenario.graph.max_lag, scenario.graph.burn_in)
        return ObservationalGT(players, B, n_context, scenario)
    raise ValueError(bg)


def _gt_value_functions(scenario, target, players, shap: ShapCfg,
                        backgrounds) -> dict[BackgroundEnum, ValueFunction]:
    predictor = GraphPredictor(scenario, target, players, shap.num_draws)
    return {
        bg: ValueFunction(
            _fill(bg, players, shap, backgrounds, clean=True,
                  scenario=scenario), predictor, shap.num_groups)
        for bg in shap.backgrounds
    }


def _model_value_functions(model, exp_cfg: ExperimentCfg, players,
                           shap: ShapCfg,
                           backgrounds) -> dict[BackgroundEnum, ValueFunction]:
    """Models only ever see noisy values, so their fills are noisy. That
    rules out OBSERVATIONAL_GT, whose fills are clean draws from the true
    graph."""
    predictor = ModelPredictor(model, exp_cfg)
    vfs = {}
    for bg in shap.backgrounds:
        if bg == BackgroundEnum.OBSERVATIONAL_GT:
            continue
        vfs[bg] = ValueFunction(
            _fill(bg, players, shap, backgrounds, clean=False, scenario=None),
            predictor, shap.num_groups)
    return vfs


# === explaining and caching ===


def _explain_all(path: Path, samples: list[SampleWithSHAP],
                 meta: SHAPFileMetadata, vfs: dict[BackgroundEnum,
                                                   ValueFunction],
                 predictor: PredictorEnum, players: list[str],
                 explain_cfg: ExplainCfg) -> None:
    """Fill in ``predictor``'s SHAP for every sample and background, saving
    each sample to ``path`` as it finishes. Results already in ``path`` are
    kept if they were computed with the same settings, on the same samples."""
    _merge_saved(path, samples, meta, predictor)
    mode = 'r+' if path.exists() else 'w'
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode) as f:
        write_metadata(f, meta)
        for i, sample in enumerate(samples):
            todo = [bg for bg in vfs if (predictor, bg) not in sample.shap_vals]
            for bg in todo:
                LOG.info(f'sample {i + 1}/{len(samples)}: {predictor.value} '
                         f'x {bg.value}')
                sample.shap_vals[(predictor, bg)] = explain_sample(
                    sample, vfs[bg], players, predictor, bg, explain_cfg)
                if vfs[bg].min_ess < 0.1 * vfs[bg].bg_fill.num_draws:
                    LOG.warning(f'low ESS {vfs[bg].min_ess:.0f}')
            write_sample(f, sample, i)


def _merge_saved(path: Path, samples: list[SampleWithSHAP],
                 meta: SHAPFileMetadata, predictor: PredictorEnum) -> None:
    """Drop ``predictor``'s results from ``samples``, then add back those
    saved at ``path`` that are still valid. If ``path`` holds results from
    different settings it is overwritten."""
    for s in samples:
        for k in [k for k in s.shap_vals if k[0] == predictor]:
            del s.shap_vals[k]
    if not path.exists():
        return
    with h5py.File(path, 'r') as f:
        saved = read_metadata(f)
        if not _compatible(saved, meta):
            LOG.info(f'{path}: different settings, recomputing {predictor}')
            stale = True
        else:
            stale = False
            for i in sample_indices(f):
                if i >= len(samples):
                    break
                old = read_sample(f, i)
                if old.seed != samples[i].seed:
                    continue  # a different sample at this index
                samples[i].shap_vals.update({
                    k: v
                    for k, v in old.shap_vals.items() if k[0] == predictor
                })
    if stale and predictor == PredictorEnum.MODEL:
        path.unlink()  # per-experiment file: start again


def _compatible(saved: SHAPFileMetadata, requested: SHAPFileMetadata) -> bool:
    """Same samples and same SHAP settings. The list of backgrounds is left
    out: adding one should compute just that one."""
    return (_same_selection(saved, requested) and
            _shap_key(saved.interpret.shap) == _shap_key(
                requested.interpret.shap) and
            saved.interpret.background == requested.interpret.background)


def _shap_key(shap: ShapCfg) -> ShapCfg:
    return replace(shap, backgrounds=[])


# === paths and datasets ===


def _shared_path(shap_path: Path, target: PredictionTarget) -> Path:
    if target.dataset_type in ('binary', 'hazard'):
        tag = f'{target.dataset_type}_h{target.horizon}'
    else:
        tag = 'survival_' + '-'.join(str(b) for b in (target.bins or ()))
    tag += f'_min{target.min_length}'
    return shap_path.parent / 'shap_samples' / f'{shap_path.stem}_{tag}.h5'


def _raw_dataset(dcfg: DatasetCfg, paths: list, features: list[str]):
    """Unnormalised dataset (``raw_features`` returns raw units)."""
    common = dict(hdf5_paths=paths,
                  feature_names=features,
                  min_length=dcfg.min_length)
    if isinstance(dcfg, BinaryDatasetCfg):
        return BinaryCellDataset(**dcfg.dataset_kwargs(),
                                 **common)
    if isinstance(dcfg, SurvivalDatasetCfg):
        return SurvivalCellDataset(num_bins=dcfg.num_bins,
                                   max_time_to_death=dcfg.max_time_to_death,
                                   event_time_bins=dcfg.bins,
                                   **common)
    raise ValueError(f'no SHAP support for {type(dcfg).__name__}')
