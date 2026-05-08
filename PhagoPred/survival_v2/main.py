"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite
from PhagoPred.survival_v2.interpret import interpret
from PhagoPred.survival_v2.utils.dataset_analysis import analyse_suite_datasets, plot_suite_event_distributions, plot_suite_absolute_distributions


def train():
    suites = (
        # 'Graph Survival',
        'Graph Binary',
        'Graph Noise Survival',
        'Graph Noise Binary',
    )
    for suite in suites:
        run_experiment_suite(
            suite_name=suite,
            output_dir='PhagoPred/survival_v2/experiments/results',
            device='cuda',
            repeats=1,
            shap_interpret=False)
    # _ = run_experiment_suite(
    #     suite_name='Graph Survival',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=1,
    #     shap_interpret=True)
    # _ = run_experiment_suite(
    #     suite_name='Graph Binary',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=3,
    #     shap_interpret=True)
    # _ = run_experiment_suite(
    #     suite_name='Graph Noise Survival',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=3,
    #     shap_interpret=True)
    # _ = run_experiment_suite(
    #     suite_name='Graph Noise Binary',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=3,
    #     shap_interpret=True)
    # # results = run_experiment_suite(
    #     suite_name='Quick Binary Test',
    #     output_dir='PhagoPred/survival_v2/experiments/results',
    #     device='cuda',
    #     repeats=2,
    # )


def eval():
    evaluate_suite(
        Path(
            'PhagoPred/survival_v2/experiments/results/Graph Binary_24042026_092301'
        ))
    evaluate_suite(
        Path(
            'PhagoPred/survival_v2/experiments/results/Graph Survival_24042026_094729'
        ))


# def interpret_suite(suite_dir: Path):
#     for experient_dir in suite_dir.iterdir():
#         if experient_dir.is_dir():
#             interpret(experiment_dir)


def view_dataset_distributions():
    """Plot event time distributions"""
    path = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Quick Survival Test_16032026_151947'
    )
    results = analyse_suite_datasets(path)
    plot_suite_event_distributions(path, results)
    plot_suite_absolute_distributions(path, results)


if __name__ == '__main__':
    # interpret(
    #     '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Survival_30042026_140226/experiment_06'
    # )
    train()
    # eval()
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
