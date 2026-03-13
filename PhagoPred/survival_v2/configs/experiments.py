from __future__ import annotations
from dataclasses import dataclass
from itertools import product

from .models import MODELS, ModelCfg
from .attention import ATTENTION, AttentionCfg
from .losses import LOSSES, LossCfg
from .datasets import DATASETS, FEATURE_COMBOS, DatasetCfg
from .training import TRAINING, TrainingCfg


@dataclass
class ExperimentCfg:
    """Dataclass to hold configurations for experiments.

    Each field accepts either a single config object or a list of config objects.
    Pass to generate_experiment_grid() to expand into one ExperimentCfg per combination.

    feature_combo convention:
        - single combo: list[str]  e.g. ['frame_count', 'linear_trend']
        - multiple combos: list[list[str]]  e.g. [['frame_count'], ['frame_count', 'linear_trend']]
    """
    model: list[ModelCfg] | ModelCfg
    attention: list[AttentionCfg] | AttentionCfg
    loss: list[LossCfg] | LossCfg
    dataset: list[DatasetCfg] | DatasetCfg
    training: list[TrainingCfg] | TrainingCfg
    feature_combo: list[list[str]] | list[str]


def generate_experiment_grid(
        experiment_cfg: ExperimentCfg) -> list[ExperimentCfg]:
    """Expand an ExperimentCfg into a list of single-valued ExperimentCfgs.

    Each field that holds a list is iterated over; the cartesian product of all
    fields is returned as individual ExperimentCfg instances.
    """

    def to_list(val):
        return val if isinstance(val, list) else [val]

    def normalize_feature_combo(val: list) -> list[list[str]]:
        """Wrap a single feature combo (list[str]) in an outer list."""
        if not val or isinstance(val[0], list):
            return val
        return [val]

    models = to_list(experiment_cfg.model)
    attentions = to_list(experiment_cfg.attention)
    losses = to_list(experiment_cfg.loss)
    datasets = to_list(experiment_cfg.dataset)
    trainings = to_list(experiment_cfg.training)
    feature_combos = normalize_feature_combo(experiment_cfg.feature_combo)

    return [
        ExperimentCfg(
            model=model,
            attention=attention,
            loss=loss,
            dataset=dataset,
            training=training,
            feature_combo=feature_combo,
        ) for model, attention,
        loss, dataset, training, feature_combo in product(
            models, attentions, losses, datasets, trainings, feature_combos)
    ]


EXPERIMENT_SUITES = {
    'Quick Survival Test':
    generate_experiment_grid(
        ExperimentCfg(model=[
            MODELS['CNN Medium'],
            MODELS['LSTM Medium'],
            MODELS['Random Forest'],
        ],
                      attention=ATTENTION['Last'],
                      loss=LOSSES['NLL'],
                      dataset=DATASETS['Survival Baseline'],
                      training=TRAINING['Quick'],
                      feature_combo=FEATURE_COMBOS['All'])),
    'Quick Binary Test':
    generate_experiment_grid(
        ExperimentCfg(model=[
            MODELS['CNN Medium'],
            MODELS['LSTM Medium'],
            MODELS['Random Forest'],
        ],
                      attention=ATTENTION['Last'],
                      loss=LOSSES['BCE'],
                      dataset=DATASETS['Binary Baseline'],
                      training=TRAINING['Quick'],
                      feature_combo=FEATURE_COMBOS['All'])),
}
# EXPERIMENT_SUITES = {
#     'quick_test':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=[MODELS['LSTM Medium'], MODELS['CNN Medium']],
#             attention=ATTENTION['Vector'],
#             loss=LOSSES['NLL'],
#             dataset=DATASETS['Survival Baseline'],
#             training=TRAINING['Quick'],
#             feature_combo=FEATURE_COMBOS['all'],
#         )),
#     'attention_comparison':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=[MODELS['LSTM Medium'], MODELS['CNN Medium']],
#             attention=list(ATTENTION.values()),
#             loss=LOSSES['Soft Target NLL'],
#             dataset=DATASETS['Survival Baseline'],
#             training=TRAINING['Standard'],
#             feature_combo=FEATURE_COMBOS['all'],
#         )),
#     'loss_comparison':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=MODELS['LSTM Medium'],
#             attention=ATTENTION['Vector'],
#             loss=[
#                 LOSSES['NLL'], LOSSES['Soft Target NLL'],
#                 LOSSES['NLL + Ranking'], LOSSES['Soft NLL + Ranking']
#             ],
#             dataset=DATASETS['Survival Baseline'],
#             training=TRAINING['Standard'],
#             feature_combo=FEATURE_COMBOS['all'],
#         )),
#     'model_comparison':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=list(MODELS.values()),
#             attention=ATTENTION['Vector'],
#             loss=LOSSES['NLL'],
#             dataset=DATASETS['Survival Baseline'],
#             training=TRAINING['Standard'],
#             feature_combo=FEATURE_COMBOS['all'],
#         )),
#     'binary_comparison':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=[MODELS['LSTM Medium'], MODELS['CNN Medium']],
#             attention=ATTENTION['Vector'],
#             loss=[LOSSES['BCE'], LOSSES['Weighted BCE'], LOSSES['Focal Loss']],
#             dataset=DATASETS['Binary Baseline'],
#             training=TRAINING['Standard'],
#             feature_combo=FEATURE_COMBOS['all'],
#         )),
#     'feature_comparison':
#     generate_experiment_grid(
#         ExperimentCfg(
#             model=MODELS['LSTM Medium'],
#             attention=ATTENTION['Vector'],
#             loss=LOSSES['NLL'],
#             dataset=DATASETS['Survival Baseline'],
#             training=TRAINING['Standard'],
#             feature_combo=[
#                 FEATURE_COMBOS['all'], FEATURE_COMBOS['no_frame_count']
#             ],
#         )),
# }
