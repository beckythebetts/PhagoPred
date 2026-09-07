"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite, interpret_suite
from PhagoPred.survival_v2.experiments.plots.plot_experiments import plot_experiment_results
from PhagoPred.survival_v2.interpret import interpret
from PhagoPred.survival_v2.interpret.ground_truth_importance import compare_importance, backfill_horizon_hazard
from PhagoPred.survival_v2.utils.dataset_analysis import analyse_suite_datasets, plot_suite_event_distributions, plot_suite_absolute_distributions


def train():
    suites = ('24_07_test', )
    for suite in suites:
        output_dir = run_experiment_suite(
            suite_name=suite,
            output_dir='PhagoPred/survival_v2/experiments/results',
            device='cuda',
            repeats=1,
            shap_interpret=True)
    return output_dir


def shap_comparison(suite_dir: Path):
    for exp_dir in suite_dir.iterdir():
        if exp_dir.is_dir():
            compare_importance(exp_dir)


def eval():
    evaluate_suite(
        Path(
            '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_01092026_095604'
        ))


def interpret_suite(suite_dir: Path):
    for experient_dir in suite_dir.iterdir():
        # print(experient_dir)
        if experient_dir.is_dir():
            interpret(experient_dir)


def view_dataset_distributions():
    """Plot event time distributions"""
    path = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Quick Survival Test_16032026_151947'
    )
    results = analyse_suite_datasets(path)
    plot_suite_event_distributions(path, results)
    plot_suite_absolute_distributions(path, results)


def plot():
    plot_experiment_results(Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/24_07_time_split_03092026_155758'
    ),
                            order_dict={
                                'dataset': ['Day 1', 'Day 2', 'Day 3']
                            })


if __name__ == '__main__':
    train()
    # plot()
