from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, fields, asdict
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from PhagoPred.survival_v2.models import ClassicalSurvivalModel, SurvivalModel
from PhagoPred.survival_v2.utils.plots import (
    visualise_binary_prediction,
    plot_roc_curve,
    plot_binary_cm,
    plot_cdf_mse_by_length,
)
from PhagoPred.survival_v2.configs.datasets import DatasetCfg
from PhagoPred.utils.logger import get_logger
from PhagoPred.utils.tools import to_json_safe
from PhagoPred.survival_v2.data import BinaryCellDataset, BinaryClassDataset, BinaryCellSample, BinaryClassCell, BinaryCell, binary_collate_fn, binary_class_collate_fn, DenseCellDataset, dense_collate_fn
from .metrics import (
    reciever_operator_characteristic,
    mean_squared_error,
)

log = get_logger()


@dataclass
class BinaryResults:
    """Data class to hold evaluation results for binary survival prediction."""
    ROC_AUC: float  # pylint: disable=invalid-name
    MSE: float  # pylint: disable=invalid-name
    n_samples: int
    n_events: int
    cm: np.ndarray
    fpr: np.ndarray | list[float]
    tpr: np.ndarray | list[float]
    cdf_mse: float | None = None

    # Exact variance of the true event probability over all valid
    # (cell, landmark) pairs of the evaluated dataset (used for plotting).
    total_true_variance: float | None = None
    unobserved_true_variance: float | None = None
    # Monte-Carlo estimate of the total variance simulated from the
    # scenario's causal graph (noisy; kept for reference).
    graph_estimated_total_variance: float | None = None

    # 'all_landmarks': every valid (cell, landmark) pair scored (exact);
    # 'sampled': random landmarks drawn with the dataset's sampler.
    evaluation_mode: str | None = None

    # cdf_mse binned by history length at the landmark (landmark - start):
    # {'bin_width', 'bin_starts', 'cdf_mse', 'n'}; cdf_mse is None for bins
    # too sparse to be reliable (see MIN_LENGTH_BIN_COUNT / _FRACTION).
    cdf_mse_by_length: dict | None = None


# Deep models without dense support are scored on this many sampler draws
# per dataset cell (with replacement), to reduce evaluation noise.
SAMPLED_EVAL_PASSES = 10
# Max points kept for the saved ROC curve (AUC is computed on the full curve).
MAX_ROC_POINTS = 1000
# Sequence-length bins for cdf_mse_by_length. Bins need at least
# MIN_LENGTH_BIN_COUNT predictions and MIN_LENGTH_BIN_FRACTION of all of them;
# this drops the sparse final bins (e.g. landmarks after end-of-recording - H,
# which only cells that die can have).
LENGTH_BIN_WIDTH = 25
MIN_LENGTH_BIN_COUNT = 100
MIN_LENGTH_BIN_FRACTION = 0.005


def evaluate_binary_model(model,
                          dataset: BinaryCellDataset,
                          save_dir: Path,
                          dataset_cfg: DatasetCfg,
                          device='cpu',
                          visualise_predicitons: int = 10,
                          batch_size: int = 128) -> BinaryResults:
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

    # if isinstance(dataset, torch.utils.data.DataLoader):
    #     data_loader = dataset
    #     dataset = data_loader.dataset

    # elif isinstance(dataset, BinaryCellDataset):
    #     data_loader = None

    # else:
    #     raise TypeError(
    #         'dataset must be torch dataloader or BinaryCellDataset')
    feature_names = dataset.feature_names

    evaluation_mode = 'sampled'
    lengths = None  # history length (landmark - start) of each prediction
    if isinstance(model, ClassicalSurvivalModel):
        predictions, labels, event_probs = _get_classical_predictions(
            model, dataset, save_dir, visualise_predicitons, feature_names)
    elif isinstance(model, SurvivalModel):
        dense_results = None
        if isinstance(dataset, BinaryCellDataset):
            dense_results = _get_dense_predictions(model, dataset, device,
                                                   batch_size)
        if dense_results is not None:
            evaluation_mode = 'all_landmarks'
            predictions, labels, event_probs, lengths = dense_results
            # Only for the example prediction plots
            _get_deep_predictions(model,
                                  dataset,
                                  device,
                                  save_dir,
                                  visualise_predicitons,
                                  feature_names,
                                  batch_size,
                                  num_samples=visualise_predicitons)
        else:
            predictions, labels, event_probs, lengths = _get_deep_predictions(
                model,
                dataset,
                device,
                save_dir,
                visualise_predicitons,
                feature_names,
                batch_size,
                num_samples=SAMPLED_EVAL_PASSES * len(dataset))
    else:
        raise ValueError(
            "Model must be an instance of SurvivalModel or ClassicalSurvivalModel."
        )
    log.info(f'Evaluating binary model on {len(labels)} samples '
             f'({evaluation_mode})')

    roc_auc, fpr, tpr, thresholds = reciever_operator_characteristic(
        predictions, labels)
    mse = mean_squared_error(predictions.ravel(), labels)

    valid = np.array([e is not None for e in event_probs], dtype=bool)
    cdf_mean_squared_error = None
    cdf_mse_by_length = None
    if valid.any():
        true_probs = np.array([e for e in event_probs if e is not None],
                              dtype=float)
        squared_errors = (predictions.ravel()[valid] - true_probs)**2
        cdf_mean_squared_error = float(np.mean(squared_errors))
        if lengths is not None:
            cdf_mse_by_length = _bin_by_length(squared_errors,
                                               np.asarray(lengths)[valid])

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    cm = confusion_matrix(labels,
                          predictions > optimal_threshold,
                          labels=[0, 1])
    fpr, tpr = _thin_curve(fpr, tpr, MAX_ROC_POINTS)

    variances_json = save_dir.parent / 'variances.json'
    cache_key = str(dataset_cfg)

    cache = {}
    if variances_json.exists():
        with variances_json.open('r') as f:
            cache = json.load(f)

    # Entries written before 'exact_total' existed stored the graph-simulated
    # estimate under 'total'; reuse it as graph_total and add the exact value.
    entry = cache.get(cache_key, {})
    if 'graph_total' not in entry and 'total' in entry:
        entry['graph_total'] = entry.pop('total')
    updated = False
    if 'exact_total' not in entry:
        exact_fn = getattr(dataset, 'get_exact_total_variance', None)
        entry['exact_total'] = exact_fn() if exact_fn is not None else None
        updated = True
    if 'graph_total' not in entry or 'unobserved' not in entry:
        entry['graph_total'] = np.asarray(
            dataset.get_total_variances()).tolist()
        entry['unobserved'] = np.asarray(
            dataset.get_unobserved_variances()).tolist()
        updated = True
    if updated:
        cache[cache_key] = entry
        with variances_json.open('w') as f:
            json.dump(cache, f, indent=2)

    total_variance = entry['exact_total']
    graph_total_variance = entry['graph_total']
    unobserved_variance = entry['unobserved']

    results = BinaryResults(
        ROC_AUC=roc_auc,
        MSE=mse,
        n_samples=len(labels),
        n_events=np.sum(labels),
        cm=cm,
        fpr=fpr,
        tpr=tpr,
        cdf_mse=cdf_mean_squared_error,
        total_true_variance=total_variance,
        unobserved_true_variance=unobserved_variance,
        graph_estimated_total_variance=graph_total_variance,
        evaluation_mode=evaluation_mode,
        cdf_mse_by_length=cdf_mse_by_length,
    )

    plot_roc_curve(fpr,
                   tpr,
                   roc_auc,
                   save_path=save_dir / "plots" / "roc_curve.png")
    plot_binary_cm(cm, save_path=save_dir / "plots" / "confusion_matrix.png")
    if cdf_mse_by_length is not None:
        plot_cdf_mse_by_length(cdf_mse_by_length,
                               total_variance,
                               unobserved_variance,
                               save_path=save_dir / "plots" /
                               "cdf_mse_by_length.png")

    results_dict = to_json_safe(asdict(results))
    log.info(f'Saving evaluation results ({results_dict}) to {save_dir}')

    with (save_dir / "evaluation_results.json").open("w") as f:
        json.dump(results_dict, f)

    return results


def _get_deep_predictions(
    model: SurvivalModel,
    dataset: BinaryCellDataset | BinaryClassDataset,
    device: str,
    save_dir: Path,
    visualise_predicitons: int = 10,
    feature_names: list[str] = None,
    batch_size: int = 128,
    num_samples: int | None = None,
):
    """Predict on landmarks drawn with the dataset's sampler (num_samples
    draws with replacement; defaults to len(dataset)).

    Returns (predictions, labels, event_probs, history_lengths)."""
    model.eval()
    model = model.to(device)
    if isinstance(dataset, BinaryCellDataset):
        collate_fn = binary_collate_fn
        target_attr = 'event'
        sample_cls = BinaryCell
    elif isinstance(dataset, BinaryClassDataset):
        collate_fn = binary_class_collate_fn
        target_attr = 'target_class'
        sample_cls = BinaryClassCell

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, device=device),
        sampler=dataset.make_sampler(num_samples))

    all_predictions = []
    all_labels = []
    all_event_probs = []
    all_lengths = []

    visualised = 0

    with torch.no_grad():
        for batch in data_loader:
            # batch: BinaryCell | ?
            features = batch.features.to(device)
            lengths = batch.length.to(device)
            mask = batch.mask
            if mask is not None:
                mask = mask.to(device)

            outputs = model(features, lengths, mask=mask)
            binary_preds = model.predict_binary(outputs).cpu().numpy()
            all_predictions.append(binary_preds)
            all_labels.append(getattr(batch, target_attr).cpu().numpy())
            all_lengths.append(lengths.cpu().numpy())

            if hasattr(batch, 'event_probability'):
                event_probs = batch.event_probability
                if event_probs is not None:
                    if hasattr(event_probs, 'cpu'):
                        event_probs = event_probs.cpu().numpy()
                    all_event_probs.extend(event_probs)
                else:
                    all_event_probs.extend([None] * len(binary_preds))
            # else:
            #     event_probs = None

            for i in range(len(lengths)):
                if visualised < visualise_predicitons:
                    single_sample_dict = {}
                    for field in fields(sample_cls):
                        single_sample_dict[field.name] = getattr(
                            batch, field.name)[i]
                    single_sample = sample_cls(**single_sample_dict)
                    visualise_binary_prediction(
                        single_sample,
                        binary_preds[i][0],
                        feature_names,
                        save_dir / 'plots' / f"prediction_{visualised}.png",
                    )
                    visualised += 1

    return np.concatenate(all_predictions), np.concatenate(
        all_labels), all_event_probs, np.concatenate(all_lengths)


def _get_dense_predictions(
    model: SurvivalModel,
    dataset: BinaryCellDataset,
    device: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, list, np.ndarray] | None:
    """Predict on every valid (cell, landmark) pair with one forward pass per
    cell. Returns (predictions, labels, event_probs, history_lengths), or None
    if the model does not support dense output (not causal / not last_pool),
    so the caller can fall back to sampling."""
    model.eval()
    model = model.to(device)
    data_loader = torch.utils.data.DataLoader(
        DenseCellDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: dense_collate_fn(batch, device=device))

    all_predictions, all_labels, all_event_probs, all_lengths = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            try:
                outputs = model.forward_dense(batch.features.to(device),
                                              batch.length.to(device),
                                              mask=batch.mask.to(device))
            except (ValueError, NotImplementedError) as e:
                log.info(f'Dense evaluation unavailable ({e}); sampling '
                         'landmarks instead')
                return None
            probs = model.predict_binary(outputs)[..., 0]
            landmark_mask = batch.dense_mask.bool()
            # timestep j predicts landmark start + j + 1, i.e. history length j + 1
            timestep = torch.arange(landmark_mask.shape[1],
                                    device=landmark_mask.device)
            all_lengths.append(
                (timestep.expand_as(landmark_mask)[landmark_mask] +
                 1).cpu().numpy())
            all_predictions.append(probs[landmark_mask].cpu().numpy())
            all_labels.append(batch.dense_event[landmark_mask].cpu().numpy())
            if batch.dense_event_probability is not None:
                all_event_probs.append(
                    batch.dense_event_probability[landmark_mask].cpu().numpy())

    predictions = np.concatenate(all_predictions)[:, np.newaxis]
    labels = np.concatenate(all_labels).astype(int)
    event_probs = (list(np.concatenate(all_event_probs))
                   if all_event_probs else [None] * len(labels))
    return predictions, labels, event_probs, np.concatenate(all_lengths)


def _bin_by_length(squared_errors: np.ndarray,
                   lengths: np.ndarray,
                   bin_width: int = LENGTH_BIN_WIDTH,
                   min_count: int = MIN_LENGTH_BIN_COUNT) -> dict:
    """Mean squared error per history-length bin."""
    lengths = lengths.astype(int)
    first = (lengths.min() // bin_width) * bin_width
    bins = (lengths - first) // bin_width
    counts = np.bincount(bins)
    sums = np.bincount(bins, weights=squared_errors)
    min_count = max(min_count, MIN_LENGTH_BIN_FRACTION * len(lengths))
    return {
        'bin_width': bin_width,
        'bin_starts': [int(first + b * bin_width) for b in range(len(counts))],
        'cdf_mse': [float(s / c) if c >= min_count else None
                    for s, c in zip(sums, counts)],
        'n': [int(c) for c in counts],
    }


def _thin_curve(fpr: np.ndarray, tpr: np.ndarray,
                max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Evenly subsample a ROC curve for saving/plotting, keeping endpoints."""
    if len(fpr) <= max_points:
        return fpr, tpr
    idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return np.asarray(fpr)[idx], np.asarray(tpr)[idx]


def _get_classical_predictions(model: ClassicalSurvivalModel,
                               dataset: BinaryCellDataset,
                               save_dir: Path,
                               visualise_predicitons: int = 10,
                               feature_names: list[str] = None):
    val_data = model.get_temporal_summary(dataset)
    all_predicitons = model.predict(val_data)
    all_labels = val_data.event.astype(int)

    event_probs = (list(val_data.event_probability)
                   if val_data.event_probability is not None else [None] *
                   len(all_predicitons))

    log.info(
        f'Got all_preeictions, shape: {all_predicitons.shape}, {type(all_predicitons)}'
    )
    visualised = 0

    for i in range(len(all_labels)):
        if visualised < visualise_predicitons:
            single_sample_dict = {}
            for field in fields(BinaryCell):
                single_sample_dict[field.name] = getattr(val_data,
                                                         field.name)[i]
            single_sample = BinaryCellSample(**single_sample_dict)
            visualise_binary_prediction(
                single_sample,
                all_predicitons[i],
                feature_names,
                save_dir / "plots" / f"prediction_{visualised}.png",
            )
            visualised += 1

    return all_predicitons, all_labels, event_probs
