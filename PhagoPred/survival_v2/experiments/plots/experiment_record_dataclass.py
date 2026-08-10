from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from PhagoPred.survival_v2.configs import ExperimentCfg
from PhagoPred.survival_v2.evaluate import BinaryResults, SurvivalResults


@dataclass
class ExperimentRecord:
    """Dataclass to hold data for one full experiment"""
    experiemnt_cfg: ExperimentCfg
    results: Union[BinaryResults, SurvivalResults]
    training_history: list[dict]
    # Needed by the SHAP plots, which read the experiment's shap_samples.h5.
    experiment_dir: Path | None = None
    # variances: dict | None = None
