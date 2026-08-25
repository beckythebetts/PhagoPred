from typing import Union
from pathlib import Path

from PhagoPred.survival_v2.models import SurvivalModel, ClassicalSurvivalModel
from PhagoPred.survival_v2.data import CellDataset, BinaryCellDataset, SurvivalCellDataset, BinaryClassDataset
from PhagoPred.survival_v2.configs.datasets import DatasetCfg
from .binary_evaluation import evaluate_binary_model, BinaryResults
from .survival_evaluation import evaluate_survival_model, SurvivalResults


def evaluate(model: Union[SurvivalModel, ClassicalSurvivalModel],
             dataset: CellDataset,
             save_dir: Path,
             dataset_cfg: DatasetCfg,
             device: str = 'cpu') -> Union[BinaryResults, SurvivalResults]:
    """Evalaute datset as either binary or survival."""
    if isinstance(dataset, (BinaryCellDataset, BinaryClassDataset)):
        results: BinaryResults = evaluate_binary_model(model, dataset,
                                                       save_dir, dataset_cfg,
                                                       device)
    elif isinstance(dataset, SurvivalCellDataset):
        results: SurvivalResults = evaluate_survival_model(
            model, dataset, save_dir, dataset_cfg, device)
    else:
        raise TypeError(
            f'Dataset for evaluation must be SurvivalCellDataset or BinaryCellDataset, not {type(dataset)}'
        )
    return results
