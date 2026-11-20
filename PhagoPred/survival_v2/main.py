"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite
from PhagoPred.survival_v2.experiments.plots.plot_experiments import plot_experiment_results
# from PhagoPred.survival_v2.interpret import interpret
# from PhagoPred.survival_v2.interpret.ground_truth_importance import compare_importance, backfill_horizon_hazard
from PhagoPred.survival_v2.interpret import run_shap
from PhagoPred.survival_v2.interpret.plots import plot_samples, plot_dataset_average


def train():
    suites = ('Graph FP Binary', )
    for suite in suites:
        output_dir = run_experiment_suite(
            suite_name=suite,
            output_dir='PhagoPred/survival_v2/experiments/results',
            device='cuda',
            repeats=1,
            shap_interpret=False,
        )
    return output_dir


# def shap_comparison(suite_dir: Path):
#     for exp_dir in suite_dir.iterdir():
#         if exp_dir.is_dir():
#             compare_importance(exp_dir)


def eval():
    evaluate_suite(
        Path(
            '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_01092026_095604'
        ))


def interpret_suite(suite_dir: Path):
    for experient_dir in suite_dir.iterdir():
        # print(experient_dir)
        if experient_dir.is_dir():
            run_shap(experient_dir)  # settings: interpret.config.INTERPRETS
            plot_samples(experient_dir / 'shap.h5',
                         experient_dir / 'shap_plots')


def plot():
    plot_experiment_results(Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_01092026_095604'
    ),
                            order_dict={
                                'dataset': ['Day 1', 'Day 2', 'Day 3']
                            })


if __name__ == '__main__':
    # train()
    suite = Path(
        '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph FP Binary_30092026_174128'
    )
    # interpret_suite(suite)
    # evaluate_suite(suite)
    plot_experiment_results(suite)
    # interpret_suite(suite)
    # train()
    # plot_experiment_results(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary Attention_24092026_104834'
    #     ))
    # interpret_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary Attention_24092026_104834'
    #     ))
    # plot_experiment_results(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary Attention_24092026_104834'
    #     ))
    # train()
    # interpret_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_01092026_095604'
    #     ))
    # plot_experiment_results(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_01092026_095604'
    #     ))
    # SHAP comparisons across the suite's varying params (plot_shap_average,
    # plot_shap_average_folds, plot_shap_sample) need interpret_suite's
    # shap_samples.h5 files to already exist.
    # plot()
