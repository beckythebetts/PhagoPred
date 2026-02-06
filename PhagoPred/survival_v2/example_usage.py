"""
Example usage of the survival_v2 framework.
Demonstrates how to run experiments programmatically.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite, evaluate_suite, interpret_model
from PhagoPred.survival_v2.interpret import SurvivalSHAP, create_background_from_dataset

def train():
    results = run_experiment_suite(
        suite_name='classical_comparison',
        output_dir='PhagoPred/survival_v2/experiments/results',
        device='cuda',
        repeats=3,
    )

def eval():
    evaluate_suite(
        Path('PhagoPred/survival_v2/experiments/results/framecount_feature_comparison_20260113_165826')
    )

def interpret():
    interpret_model(
        Path('/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/attention_comparison_20260127_174046/cnn_medium_none_last_soft_target_late_entry_extreme_standard_all_00')
    )
if __name__ == '__main__':
    train()
    # interpret()
    # results = run_experiment_suite(
    #     suite_name=
    # )
    
    # evaluate_suite(
    #     Path('/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/framecount_feature_comparison_20260113_165826')
    # )
    

