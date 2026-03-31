"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite
from PhagoPred.survival_v2.interpret import SurvivalSHAP, create_background_from_dataset
from PhagoPred.survival_v2.utils.dataset_analysis import analyse_suite_datasets, plot_suite_event_distributions, plot_suite_absolute_distributions


def train():
    # run_experiment_suite(
    #     suite_name='SurvivalTest',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=5,
    # )
    run_experiment_suite(
        suite_name='BinaryTest',
        output_dir='PhagoPred/survival_v2/experiments/results',
        device='cuda',
        repeats=5,
    )
    # # results = run_experiment_suite(
    #     suite_name='Quick Binary Test',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=2,
    # )


# def eval():
#     evaluate_suite(
#         Path(
#             'PhagoPred/survival_v2/experiments/results/framecount_feature_comparison_20260113_165826'
#         ))

# def interpret():
#     interpret_model(
#         Path(
#             'PhagoPred/survival_v2/experiments/results/classical_comparison_20260210_195114/cnn_medium_none_last_soft_target_baseline_standard_all_00'
#         ))


def view_dataset_distributions():
    """Plot event time distributions"""
    path = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Quick Survival Test_16032026_151947'
    )
    results = analyse_suite_datasets(path)
    plot_suite_event_distributions(path, results)
    plot_suite_absolute_distributions(path, results)


if __name__ == '__main__':
    train()

    # evaluate_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/BinaryTest_24032026_144422'
    #     ))
    # interpret()
    # results = run_experiment_suite(
    #     suite_name=
    # )
    # view_dataset_distributions()
    # evaluate_suite(
    #     Path('/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/framecount_feature_comparison_20260113_165826')
    # )
