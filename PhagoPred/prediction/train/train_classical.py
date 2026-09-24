from PhagoPred.survival_v2.models import ClassicalSurvivalModel
from PhagoPred.survival_v2.data import CellDataset


def train_classical(model: ClassicalSurvivalModel,
                    train_dataset: CellDataset) -> dict:
    """Fit a classicla model"""
    history = model.fit(train_dataset)
    return history
