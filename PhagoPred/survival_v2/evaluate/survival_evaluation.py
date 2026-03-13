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
from PhagoPred.utils.tools import to_json_safe
from PhagoPred.survival_v2.data import SurvivalCellDataset, survival_collate_fn, SurvivalCell, SurvivalCellSample
from .metrics import (
    concordance_index,
    integrated_brier_score,
)


@dataclass
class SurvivalResults:
    """Data class to hold evaluation results for survival prediction."""
    c_index: float
    brier_score: float
    n_samples: int
    n_events: int
    n_censored: int
    cm: np.ndarray


def evaluate_survival_model(model,
                            dataset: SurvivalCellDataset,
                            save_dir: Path,
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
    os.mkdir(save_dir / 'plots')

    feature_names = dataset.feature_names

    if isinstance(model, ClassicalSurvivalModel):
        predictions, binned_times, times, events = _get_classical_predictions(
            model, dataset, save_dir / 'plots', visulaise_predictions,
            feature_names)
    elif isinstance(model, SurvivalModel):
        predictions, binned_times, times, events = _get_deep_predictions(
            model, dataset, device, save_dir / 'plots', visulaise_predictions,
            feature_names, batch_size)
    else:
        raise ValueError(
            "Model must be an instance of SurvivalModel or ClassicalSurvivalModel."
        )

    c_index = concordance_index(predictions, times, events,
                                dataset.event_time_bins)
    ibs, brier_times, brier_scores = integrated_brier_score(
        predictions, times, events, dataset.event_time_bins)

    expected_times = (predictions * np.arange(model.num_bins)).sum(axis=1)
    cm = confusion_matrix(binned_times, expected_times.round().astype(int))

    plot_cm(cm, save_dir / "plots" / "confusion_matrix.png")
    plot_brier_scores(brier_times, brier_scores,
                      save_dir / "plots" / "brier_scores.png")

    results = SurvivalResults(
        c_index=c_index,
        brier_score=ibs,
        n_samples=len(times),
        n_events=events.sum(),
        n_censored=(1 - events).sum(),
        cm=cm,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                    )
                    visualised += 1

    return np.concatenate(all_pmfs), np.concatenate(
        all_true_bins), np.concatenate(all_true_times), np.concatenate(
            all_events)


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
            )
            visualised += 1

    return all_pmfs, all_true_bins, all_true_times, all_events
