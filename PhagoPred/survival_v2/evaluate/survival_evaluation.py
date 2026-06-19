from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from PhagoPred.survival_v2.models import ClassicalSurvivalModel, SurvivalModel
from PhagoPred.survival_v2.utils.plots import (
    visualise_survival_prediction,
    plot_cm,
    plot_brier_scores,
)
from PhagoPred.survival_v2.configs.datasets import DatasetCfg
from PhagoPred.utils.tools import to_json_safe
from PhagoPred.survival_v2.data import SurvivalCellDataset, survival_collate_fn, SurvivalCell, SurvivalCellSample
from .metrics import (
    concordance_index,
    integrated_brier_score,
    hazard_mse,
    pmf_mse,
    soft_confusion_matrix,
)


@dataclass
class SurvivalResults:
    """Data class to hold evaluation results for survival prediction."""
    C_Index: float
    Brier_Score: float
    c_index_per_bin: np.ndarray
    n_samples: int
    n_events: int
    n_censored: int
    cm_expected: np.ndarray
    cm_argmax: np.ndarray
    cm_soft: np.ndarray
    brier_times: np.ndarray
    brier_scores: np.ndarray

    hazard_mse_per_bin: np.ndarray | None = None
    pmf_mse_per_bin: np.ndarray | None = None

    total_true_hazard_variance: np.ndarray | None = None
    unobserved_true_hazard_variance: np.ndarray | None = None

    total_true_pmf_variance: np.ndarray | None = None
    unobserved_true_pmf_variance: np.ndarray | None = None


def _variance_cache_key(dataset_cfg: DatasetCfg) -> str:
    """Cache key from the dataset config; shared across a suite's experiments.

    Keep derived/fitted fields (means, stds, bins, variances) out of the cfg's
    repr (field(repr=False)) so this key stays stable.
    """
    return str(dataset_cfg)


def _get_or_compute_variances(dataset: SurvivalCellDataset,
                              dataset_cfg: DatasetCfg,
                              cache_dir: Path | None) -> tuple[dict, dict]:
    """Return (total, unobserved) variance dicts {'hazard','pmf'}, cached per suite.

    Avoids re-running the Monte-Carlo variance estimate for every experiment
    (and the previous four-calls-per-eval duplication).
    """
    path = Path(
        cache_dir) / 'variances.json' if cache_dir is not None else None
    key = _variance_cache_key(dataset_cfg)

    if path is not None and path.exists():
        with path.open('r') as f:
            entry = json.load(f).get(key)
        if entry is not None:
            return entry['total'], entry['unobserved']

    total = to_json_safe(dataset.get_total_variances())
    unobserved = to_json_safe(dataset.get_unobserved_variances())

    if path is not None:
        store = {}
        if path.exists():
            with path.open('r') as f:
                store = json.load(f)
        store[key] = {'total': total, 'unobserved': unobserved}
        with path.open('w') as f:
            json.dump(store, f, indent=2)

    return total, unobserved


def evaluate_survival_model(model,
                            dataset: SurvivalCellDataset,
                            save_dir: Path,
                            dataset_cfg: DatasetCfg,
                            device: str = 'cpu',
                            visulaise_predictions: int = 10,
                            batch_size: int = 128) -> SurvivalResults:
    """
    Evaluate a trained model on a test dataset and save visualisations of predictions.

    Args:
        model: trained SurvivalModel instance
        data_loader: DataLoader for the test dataset
        device: torch device
        visulaise_predicitons: number of predictions to visualise and save
        save_dir: directory to save visualisations
    """
    os.makedirs(save_dir / 'plots', exist_ok=True)

    feature_names = dataset.feature_names

    if isinstance(model, ClassicalSurvivalModel):
        predictions, binned_times, times, events, true_binned_pmfs = _get_classical_predictions(
            model, dataset, save_dir / 'plots', visulaise_predictions,
            feature_names)
    elif isinstance(model, SurvivalModel):
        predictions, binned_times, times, events, true_binned_pmfs = _get_deep_predictions(
            model, dataset, device, save_dir / 'plots', visulaise_predictions,
            feature_names, batch_size)
    else:
        raise ValueError(
            "Model must be an instance of SurvivalModel or ClassicalSurvivalModel."
        )

    c_indexs = concordance_index(predictions, times, events,
                                 dataset.event_time_bins)
    ibs, brier_times, brier_scores = integrated_brier_score(
        predictions, times, events, dataset.event_time_bins)

    hazard_mean_squared_error = hazard_mse(predictions, true_binned_pmfs)
    pmf_mean_squared_error = pmf_mse(predictions, true_binned_pmfs)

    # expected_times = (predictions * np.arange(model.num_bins)).sum(axis=1)
    # Only plot uncensored samples in CMs

    # pmf_sum = predictions.sum(axis=1, keepdims=True).clip(min=1e-8)
    # expected_times = (predictions / pmf_sum *
    #                   np.arange(model.num_bins)).sum(axis=1)
    # argmax_times = np.argmax(predictions, axis=1)

    # binned_times = binned_times[events == 1]
    # expected_times = expected_times[events == 1]
    # argmax_times = argmax_times[events == 1]

    # cm_expected = confusion_matrix(binned_times,
    #                                expected_times.round().astype(int))
    # cm_argmax = confusion_matrix(binned_times,
    #                              argmax_times.round().astype(int))

    # For expected time use restricted mean_survival_time
    # Area under survival curve up to final bin (N bins)$ = \sum_{i=0}^{N}(1 - CIF(i)$)
    # Add final bin rperesenting left over pmf
    cif = np.cumsum(predictions, axis=1)
    rmst = np.sum(1 - cif, axis=1)
    argmax_times = np.argmax(np.concatenate(
        (predictions, 1 - np.sum(predictions, axis=1, keepdims=True)), axis=1),
                             axis=1)

    # print(rmst.round().astype(int),
    #       argmax_times.round().astype(int), binned_times)

    cm_expected = confusion_matrix(binned_times, rmst.round().astype(int))
    cm_argmax = confusion_matrix(binned_times,
                                 argmax_times.round().astype(int))
    # Soft CM: keep the full predicted PMF instead of collapsing to one bin.
    # Raw mass counts here; plot_cm row-normalises for display like the others.
    cm_soft = soft_confusion_matrix(predictions, binned_times)

    # num_bins = dataset.num_bins
    # binned_times[events == 0] = num_bins

    plot_cm(cm_expected,
            save_dir / "plots" / "confusion_matrix_expected_times.png")
    plot_cm(cm_argmax,
            save_dir / "plots" / "confusion_matrix_argmax_times.png")
    plot_cm(cm_soft, save_dir / "plots" / "soft_confusion_matrix.png")
    plot_brier_scores(brier_times, brier_scores,
                      save_dir / "plots" / "brier_scores.png")

    # Total (direct multi-pass) and unobserved (graph) variances, computed in
    # the dataset's _load_true_variances and cached per data config.
    total_variances, unobserved_variances = _get_or_compute_variances(
        dataset, dataset_cfg, save_dir.parent)
    total_hazard_variances = total_variances['hazard']
    unobserved_hazard_variances = unobserved_variances['hazard']
    total_pmf_variances = total_variances['pmf']
    unobserved_pmf_variances = unobserved_variances['pmf']

    # hazard_mse_per_bin = _compute_hazard_mse_per_bin(predictions,
    #                                                  true_binned_pmfs)

    results = SurvivalResults(
        C_Index=c_indexs[-1],
        Brier_Score=ibs,
        c_index_per_bin=c_indexs,
        n_samples=len(times),
        n_events=events.sum(),
        n_censored=(1 - events).sum(),
        cm_expected=cm_expected,
        cm_argmax=cm_argmax,
        cm_soft=cm_soft,
        brier_times=brier_times,
        brier_scores=brier_scores,
        hazard_mse_per_bin=hazard_mean_squared_error,
        pmf_mse_per_bin=pmf_mean_squared_error,
        total_true_hazard_variance=total_hazard_variances,
        unobserved_true_hazard_variance=unobserved_hazard_variances,
        total_true_pmf_variance=total_pmf_variances,
        unobserved_true_pmf_variance=unobserved_pmf_variances,
    )

    results_dict = to_json_safe(asdict(results))

    with (save_dir / "evaluation_results.json").open('w') as f:
        json.dump(results_dict, f)

    return results


def _get_deep_predictions(
    model: SurvivalModel,
    dataset: SurvivalCellDataset,
    device: str,
    save_dir: Path,
    visualise_predicitons: int = 10,
    feature_names: list[str] = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    model.eval()
    model = model.to(device)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: survival_collate_fn(batch, device=device))

    bin_edges = dataset.event_time_bins
    all_pmfs = []
    all_true_bins = []
    all_true_times = []
    all_events = []
    all_true_binned_pmfs = []
    visualised = 0

    with torch.no_grad():
        for batch in data_loader:
            batch: SurvivalCell
            features = batch.features.to(device)
            lengths = batch.length.to(device)
            mask = batch.mask.to(device)
            if mask is not None:
                mask = mask.to(device)

            outputs = model(features, lengths, mask=mask)
            pmf = model.predict_pmf(outputs).cpu().numpy()

            all_true_bins.append(batch.time_to_event_bin.cpu().numpy())
            all_events.append(batch.event_indicator.cpu().numpy())
            all_pmfs.append(pmf)
            all_true_times.append(batch.time_to_event)

            binned_pmfs = batch.binned_pmf
            if binned_pmfs is not None:
                if hasattr(binned_pmfs, 'cpu'):
                    binned_pmfs = binned_pmfs.cpu().numpy()
                all_true_binned_pmfs.extend(binned_pmfs)
            else:
                all_true_binned_pmfs.extend([None] * len(pmf))

            for i in range(len(lengths)):
                if visualised < visualise_predicitons:
                    single_sample_dict = {}
                    for field in fields(SurvivalCell):
                        single_sample_dict[field.name] = getattr(
                            batch, field.name)[i]
                    single_sample = SurvivalCellSample(**single_sample_dict)
                    visualise_survival_prediction(
                        single_sample,
                        pmf[i],
                        bin_edges,
                        feature_names=feature_names,
                        save_path=save_dir / f"prediction_{visualised}.png",
                        padded='end',
                    )
                    visualised += 1

    return (np.concatenate(all_pmfs), np.concatenate(all_true_bins),
            np.concatenate(all_true_times), np.concatenate(all_events),
            all_true_binned_pmfs)


def _get_classical_predictions(model: ClassicalSurvivalModel,
                               dataset: SurvivalCellDataset,
                               save_dir: Path,
                               visualise_predicitons: int = 10,
                               feature_names: list[str] = None):
    val_data = model.get_temporal_summary(dataset)
    all_pmfs = model.predict(val_data)
    all_true_bins = val_data.time_to_event_bin.astype(int)
    all_true_times = val_data.time_to_event
    all_events = val_data.event_indicator.astype(int)

    true_binned_pmfs = (list(val_data.binned_pmf) if val_data.binned_pmf
                        is not None else [None] * len(all_pmfs))
    visualised = 0

    for i, pmf in enumerate(all_pmfs):
        if visualised < visualise_predicitons:
            single_sample_dict = {}
            for field in fields(SurvivalCell):
                single_sample_dict[field.name] = getattr(val_data,
                                                         field.name)[i]
            single_sample = SurvivalCellSample(**single_sample_dict)
            visualise_survival_prediction(
                single_sample,
                pmf,
                bin_edges=dataset.event_time_bins,
                feature_names=feature_names,
                save_path=save_dir / f"prediction_{visualised}.png",
                padded='none',
            )
            visualised += 1

    return all_pmfs, all_true_bins, all_true_times, all_events, true_binned_pmfs
