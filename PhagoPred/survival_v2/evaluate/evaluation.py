from typing import Union
from pathlib import Path

from PhagoPred.survival_v2.models import SurvivalModel, ClassicalSurvivalModel
from PhagoPred.survival_v2.data import CellDataset, BinaryCellDataset, SurvivalCellDataset
from .binary_evaluation import evaluate_binary_model, BinaryResults
from .survival_evaluation import evaluate_survival_model, SurvivalResults


def evaluate(model: Union[SurvivalModel, ClassicalSurvivalModel],
             dataset: CellDataset,
             save_dir: Path,
             device: str = 'cpu') -> Union[BinaryResults, SurvivalResults]:
    """Evalaute datset as either binary or survival."""
    if isinstance(dataset, BinaryCellDataset):
        results: BinaryCellDataset = evaluate_binary_model(
            model, dataset, save_dir)
    elif isinstance(dataset, SurvivalCellDataset):
        results: SurvivalCellDataset = evaluate_survival_model(
            model, dataset, save_dir, device)
    else:
        raise TypeError(
            f'Dataset for evaluation must be SurvivalCellDataset or BinaryCellDataset, not {type(dataset)}'
        )
    return results
