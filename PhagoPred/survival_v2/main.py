"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite, test_variances
from PhagoPred.survival_v2.experiments.plots.plot_experiments import plot_experiment_results
from PhagoPred.survival_v2.interpret import interpret
from PhagoPred.survival_v2.interpret.ground_truth_importance import compare_importance, backfill_horizon_hazard
from PhagoPred.survival_v2.utils.dataset_analysis import analyse_suite_datasets, plot_suite_event_distributions, plot_suite_absolute_distributions


def train():
    suites = (
        # 'Graph Survival',
        # 'Graph Binary',
        # 'Graph Noise Survival',
        # 'Graph Noise Binary',
        # 'Learning Curve Survival',
        # 'Graph Scenario Types Binary',
        # 'Graph Nonlinear Chain AR Binary',
        '24_07_test', )
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


def shap_comparison(suite_dir: Path):
    for exp_dir in suite_dir.iterdir():
        if exp_dir.is_dir():
            compare_importance(exp_dir)


def eval():
    evaluate_suite(
        Path(
            '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/24_07_test_13082026_133851'
        ))
    # evaluate_suite(
    #     Path(
    #         'PhagoPred/survival_v2/experiments/results/Graph Survival_24042026_094729'
    #     ))


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
    plot_experiment_results(
        Path(
            '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Nonlinear Chain AR Binary_23072026_173331'
        ))


if __name__ == '__main__':
    # plot()
    # train()
    eval()

    # for h5_file in Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/Datasets/graph_synthetic/shap_samples'
    # ).iterdir():
    #     backfill_horizon_hazard(
    #         h5_file,
    #         Path('/home/ubuntu/PhagoPred/PhagoPred/Datasets/graph_synthetic'),
    #         force=True)
    # shap_comparison(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Binary_07072026_164403'
    #     ))

    # plot()
    # compare_importance(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Binary_29062026_215207_high_auto_corr/experiment_05'
    #     ),
    #     n_samples=5,
    #     num_permutations=500,
    #     nsamples_shap="auto",
    #     num_segments=50,
    #     show_feature_values=True,
    # )
    # shap_comparison(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Nonlinear Chain AR Binary_23072026_173331'
    #     ))
    # shap_comparison(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Scenario Types Binary_23072026_143610'
    #     ))
    # shap_comparison(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Binary_29062026_215207_high_auto_corr'
    #     ))
    # interpret_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Binary_02072026_085002'
    #     ), )
    # compare_importance(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Learning Curve Survival_29062026_134821/experiment_00'
    #     ),
    #     n_samples=10,
    #     num_permutations=1000,
    #     nsamples_shap="auto",
    #     show_variance_bounds=True,
    #     show_feature_totals=False,
    #     # show_feature_values=True,
    #     #    show_base_noise=True,
    #     num_segments=100)
    # compare_importance(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Binary_26062026_142217/experiment_01'
    #     ),
    #     n_samples=10,
    #     num_permutations=500,
    #     nsamples_shap=100000,
    #     show_variance_bounds=True,
    #     # show_feature_values=True,
    #     #    show_base_noise=True,
    #     show_feature_totals=False,
    #     num_segments=100)
    # compare_importance(
    #     '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Survival_29062026_162358_high_auto_corr/experiment_06',
    #     n_samples=5,
    #     num_permutations=500,
    #     nsamples_shap="auto",
    #     num_segments=50,
    #     show_feature_values=True,
    #     show_conditional=True)

    # train()
    # eval()
    # plot_experiment_results(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Learning Curve Survival_11062026_121151'
    #     ))
    # plot()
    # eval()
    # evaluate_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Survival_29052026_200757'
    #     ))
    # evaluate_suite(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Survival_02062026_120922'
    #     ))

    # test_variances(
    #     Path(
    #         '/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/Graph Survival_29052026_200757/experiment_00'
    #     ))
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
