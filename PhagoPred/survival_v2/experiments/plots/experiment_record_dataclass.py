from dataclasses import dataclass
from typing import Union

from PhagoPred.survival_v2.configs import ExperimentCfg
from PhagoPred.survival_v2.evaluate import BinaryResults, SurvivalResults


@dataclass
class ExperimentRecord:
    """Dataclass to hold data for one full experiment"""
    experiemnt_cfg: ExperimentCfg
    results: Union[BinaryResults, SurvivalResults]
    training_history: list[dict]
