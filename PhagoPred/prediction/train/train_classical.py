from PhagoPred.prediction.models import ClassicalSurvivalModel
from PhagoPred.prediction.data import CellDataset


def train_classical(model: ClassicalSurvivalModel,
                    train_dataset: CellDataset) -> dict:
    """Fit a classicla model"""
    history = model.fit(train_dataset)
    return history
