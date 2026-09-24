from __future__ import annotations

from PhagoPred.survival_v2.configs.models import LSTMCfg, CNNCfg, RSFCfg
from PhagoPred.survival_v2.configs.experiments import ModelCfg, AttentionCfg
from .lstm_survival import LSTMSurvival
from .cnn_survival import CNNSurvival
from .random_forest import RandomSurvivalForestModel
from .base import SurvivalModel
from .classical_base import ClassicalSurvivalModel


def build_model(
    model_config: ModelCfg,
    attention_config: AttentionCfg,
    num_features: int,
    num_bins: int,
    device: str,
    feature_names: list[str] = None,
) -> SurvivalModel | ClassicalSurvivalModel:
    """Build a model from configuration."""
    if isinstance(model_config, LSTMCfg):
        model = LSTMSurvival(
            num_features=num_features,
            num_bins=num_bins,
            lstm_hidden_size=model_config.hidden_size,
            lstm_num_layers=model_config.num_layers,
            lstm_dropout=model_config.dropout,
            fc_layers=model_config.fc_layers,
            attention_config=attention_config,
            predictor_layers=model_config.predictor_layers,
        )
        return model.to(device)

    if isinstance(model_config, CNNCfg):
        model = CNNSurvival(
            num_features=num_features,
            num_bins=num_bins,
            num_channels=model_config.num_channels,
            kernel_sizes=model_config.kernel_sizes,
            dilations=model_config.dilations,
            fc_layers=model_config.fc_layers,
            attention_config=attention_config,
        )
        return model.to(device)

    if isinstance(model_config, RSFCfg):
        return RandomSurvivalForestModel(
            num_bins=num_bins,
            n_estimators=model_config.n_estimators,
            max_depth=model_config.max_depth,
            feature_names=feature_names,
            window_sizes=model_config.window_sizes,
        )

    raise ValueError(f"Unknown model config type: {type(model_config)}")
