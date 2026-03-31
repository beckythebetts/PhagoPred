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
)
from PhagoPred.utils.logger import get_logger
from PhagoPred.utils.tools import to_json_safe
from PhagoPred.survival_v2.data import BinaryCellDataset, BinaryCellSample, BinaryCell, binary_collate_fn
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


def evaluate_binary_model(model,
                          dataset: BinaryCellDataset,
                          save_dir: Path,
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
    os.mkdir(save_dir / 'plots')

    # if isinstance(dataset, torch.utils.data.DataLoader):
    #     data_loader = dataset
    #     dataset = data_loader.dataset

    # elif isinstance(dataset, BinaryCellDataset):
    #     data_loader = None

    # else:
    #     raise TypeError(
    #         'dataset must be torch dataloader or BinaryCellDataset')
    feature_names = dataset.feature_names

    if isinstance(model, ClassicalSurvivalModel):
        predictions, labels = _get_classical_predictions(
            model, dataset, save_dir, visualise_predicitons, feature_names)
    elif isinstance(model, SurvivalModel):
        predictions, labels = _get_deep_predictions(model, dataset, device,
                                                    save_dir,
                                                    visualise_predicitons,
                                                    feature_names, batch_size)
    else:
        raise ValueError(
            "Model must be an instance of SurvivalModel or ClassicalSurvivalModel."
        )

    roc_auc, fpr, tpr, thresholds = reciever_operator_characteristic(
        predictions, labels)
    mse = mean_squared_error(predictions, labels)

    optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    cm = confusion_matrix(labels,
                          predictions > optimal_threshold,
                          labels=[0, 1])

    results = BinaryResults(ROC_AUC=roc_auc,
                            MSE=mse,
                            n_samples=len(labels),
                            n_events=np.sum(labels),
                            cm=cm,
                            fpr=fpr,
                            tpr=tpr)

    plot_roc_curve(fpr,
                   tpr,
                   roc_auc,
                   save_path=save_dir / "plots" / "roc_curve.png")
    plot_binary_cm(cm, save_path=save_dir / "plots" / "confusion_matrix.png")

    results_dict = to_json_safe(asdict(results))
    log.info(f'Saving evaluation results ({results_dict}) to {save_dir}')

    with (save_dir / "evaluation_results.json").open("w") as f:
        json.dump(results_dict, f)

    return results


def _get_deep_predictions(
    model: SurvivalModel,
    dataset: BinaryCellDataset,
    device: str,
    save_dir: Path,
    visualise_predicitons: int = 10,
    feature_names: list[str] = None,
    batch_size: int = 128,
):
    model.eval()
    model = model.to(device)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: binary_collate_fn(batch, device=device))

    all_predictions = []
    all_labels = []

    visualised = 0

    with torch.no_grad():
        for batch in data_loader:
            batch: BinaryCell
            features = batch.features.to(device)
            lengths = batch.length.to(device)
            mask = batch.mask
            if mask is not None:
                mask = mask.to(device)

            outputs = model(features, lengths, mask=mask)
            binary_preds = model.predict_binary(outputs).cpu().numpy()
            all_predictions.append(binary_preds)
            all_labels.append(batch.event.cpu().numpy())

            for i in range(len(lengths)):
                if visualised < visualise_predicitons:
                    single_sample_dict = {}
                    for field in fields(BinaryCell):
                        single_sample_dict[field.name] = getattr(
                            batch, field.name)[i]
                    single_sample = BinaryCellSample(**single_sample_dict)
                    visualise_binary_prediction(
                        single_sample,
                        binary_preds[i][0],
                        feature_names,
                        save_dir / 'plots' / f"prediction_{visualised}.png",
                    )
                    visualised += 1

    return np.concatenate(all_predictions), np.concatenate(all_labels)


def _get_classical_predictions(model: ClassicalSurvivalModel,
                               dataset: BinaryCellDataset,
                               save_dir: Path,
                               visualise_predicitons: int = 10,
                               feature_names: list[str] = None):
    val_data = model.get_temporal_summary(dataset)
    all_predicitons = model.predict(val_data)
    all_labels = val_data.event.astype(int)

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

    return all_predicitons, all_labels
