"""
Experiment runner for survival analysis models.
Supports running multiple experiments with different configurations.
"""
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PhagoPred.survival_v2.configs.experiment_config import (
    MODELS, ATTENTION, LOSSES, DATASETS, TRAINING, EXPERIMENT_SUITES, FEATURE_COMBOS
)
from PhagoPred.survival_v2.models.lstm_survival import LSTMSurvival
from PhagoPred.survival_v2.models.cnn_survival import CNNSurvival
from PhagoPred.survival_v2.models.random_forest import RandomForestSurvival
from PhagoPred.survival_v2.models.gradient_boosting import XGBoostSurvival, LightGBMSurvival
from PhagoPred.survival_v2.models.survival_forest import RandomSurvivalForestModel, GradientBoostingSurvivalModel
from PhagoPred.survival_v2.models.classical_base import ClassicalSurvivalModel
from PhagoPred.survival_v2.losses.survival_losses import compute_loss, LOSS_CONFIGS
from PhagoPred.survival_v2.utils.training import train_model
from PhagoPred.survival_v2.utils.evaluation import evaluate_model, evaluate_classical_model
from PhagoPred.survival_v2.data.dataset import CellDataset, collate_fn
from PhagoPred.survival_v2.utils.plots import plot_losses
from PhagoPred.survival_v2.experiments.plot_experiments import plot_experiment_results, plot_confusion_matrices, plot_experiment_losses
from PhagoPred.survival_v2.interpret import SurvivalSHAP, KernelSHAP, create_background_from_dataset

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  # converts np.float32 → float
    else:
        return obj
    
def build_model(model_config, attention_config, num_features, num_bins, device, feature_names=None):
    """
    Build a model from configuration.

    Args:
        model_config: dict with model architecture parameters
        attention_config: dict with attention mechanism parameters
        num_features: number of input features
        num_bins: number of time bins
        device: torch device
        feature_names: optional list of feature names (for classical models)

    Returns:
        model: SurvivalModel or ClassicalSurvivalModel instance
    """
    model_type = model_config['type']

    # Deep learning models
    if model_type == 'lstm':
        model = LSTMSurvival(
            num_features=num_features,
            num_bins=num_bins,
            lstm_hidden_size=model_config['lstm_hidden_size'],
            lstm_num_layers=model_config['lstm_num_layers'],
            lstm_dropout=model_config['lstm_dropout'],
            fc_layers=model_config['fc_layers'],
            attention_type=attention_config['type'],
            attention_config=attention_config,
            use_predictor=model_config.get('use_predictor', False),
            predictor_layers=model_config.get('predictor_layers', [64])
        )
        return model.to(device)

    elif model_type == 'cnn':
        model = CNNSurvival(
            num_features=num_features,
            num_bins=num_bins,
            num_channels=model_config['num_channels'],
            kernel_sizes=model_config['kernel_sizes'],
            dilations=model_config['dilations'],
            fc_layers=model_config['fc_layers'],
            attention_type=attention_config['type'],
            attention_config=attention_config
        )
        return model.to(device)

    # Classical ML models
    elif model_type == 'random_forest':
        model = RandomForestSurvival(
            num_bins=num_bins,
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            feature_extraction=model_config.get('feature_extraction', 'temporal_full'),
            n_segments=model_config.get('n_segments', 5),
            n_lags=model_config.get('n_lags', 10),
            feature_names=feature_names,
            window_sizes=model_config.get('window_sizes', [10, 20, 50]),
        )
        return model

    elif model_type == 'xgboost':
        model = XGBoostSurvival(
            num_bins=num_bins,
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 6),
            learning_rate=model_config.get('learning_rate', 0.1),
            feature_extraction=model_config.get('feature_extraction', 'temporal_full'),
            n_segments=model_config.get('n_segments', 5),
            n_lags=model_config.get('n_lags', 10),
            feature_names=feature_names
        )
        return model

    elif model_type == 'lightgbm':
        model = LightGBMSurvival(
            num_bins=num_bins,
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', -1),
            num_leaves=model_config.get('num_leaves', 31),
            learning_rate=model_config.get('learning_rate', 0.1),
            feature_extraction=model_config.get('feature_extraction', 'temporal_full'),
            n_segments=model_config.get('n_segments', 5),
            n_lags=model_config.get('n_lags', 10),
            feature_names=feature_names
        )
        return model

    elif model_type == 'random_survival_forest':
        model = RandomSurvivalForestModel(
            num_bins=num_bins,
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            feature_extraction=model_config.get('feature_extraction', 'temporal_full'),
            n_segments=model_config.get('n_segments', 5),
            n_lags=model_config.get('n_lags', 10),
            feature_names=feature_names,
            window_sizes=model_config.get('window_sizes', [10, 20, 50]),
        )
        return model

    elif model_type == 'gradient_boosting_survival':
        model = GradientBoostingSurvivalModel(
            num_bins=num_bins,
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 3),
            learning_rate=model_config.get('learning_rate', 0.1),
            feature_extraction=model_config.get('feature_extraction', 'temporal_full'),
            n_segments=model_config.get('n_segments', 5),
            n_lags=model_config.get('n_lags', 10),
            feature_names=feature_names
        )
        return model

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def is_classical_model(model_config):
    """Check if model configuration is for a classical ML model."""
    classical_types = [
        'random_forest', 'xgboost', 'lightgbm',
        'random_survival_forest', 'gradient_boosting_survival'
    ]
    return model_config['type'] in classical_types


def build_datasets(dataset_config: dict, feature_combo: list, is_classical: bool=False):
    """
    Build train and validation datasets.

    Args:
        dataset_config: dict with dataset parameters
        feature_combo: list of feature names to include
        is_classical: If false, normalise dataset and don't use fixed length

    Returns:
        train_dataset: CellDataset for training
        val_dataset: CellDataset for validation
    """
    # print('DATASET CONFIG:', dataset_config)
    # model = dataset_config.get('model', None)
    if is_classical:
        normalise = False
        fixed_length = 1000
    else:
        normalise = False
        fixed_length = None
        
    # Build training dataset
    train_dataset = CellDataset(
        hdf5_paths=dataset_config['train_paths'],
        features=feature_combo,
        num_bins=dataset_config['num_bins'],
        min_length=dataset_config.get('min_length', 500),
        max_time_to_death=dataset_config.get('max_time_to_death', 100),
        preload_data=True,
        interpolate_nan=False,
        add_start_frame_feature=dataset_config.get('add_start_frame_feature', False),
        normalise=normalise,
        fixed_len=fixed_length,
    )

    # Compute normalization stats from training data
    means, stds = train_dataset.get_normalization_stats()

    # Build validation dataset with same normalization
    val_dataset = CellDataset(
        hdf5_paths=dataset_config['val_paths'],
        features=feature_combo,
        num_bins=dataset_config['num_bins'],
        min_length=dataset_config.get('min_length', 500),
        max_time_to_death=dataset_config.get('max_time_to_death', 100),
        means=means,
        stds=stds,
        event_time_bins=train_dataset.event_time_bins,
        preload_data=True,
        interpolate_nan=False,
        add_start_frame_feature=dataset_config.get('add_start_frame_feature', False),
        normalise=normalise,
        fixed_len=fixed_length,
    )

    return train_dataset, val_dataset


def run_single_experiment(
    experiment_config,
    output_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    """
    Run a single experiment.

    Args:
        experiment_config: dict with experiment configuration
        output_dir: path to save results
        device: torch device

    Returns:
        results: dict with experiment results
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_config['name']}")
    print(f"{'='*80}")

    # Create output directory
    file_idx = 0
    exp_dir = Path(output_dir) / f"{experiment_config['name']}_{file_idx:02d}"
    while exp_dir.exists():
        file_idx += 1
        exp_dir = Path(output_dir) / f"{experiment_config['name']}_{file_idx:02d}"
    exp_dir.mkdir(parents=True)

    # Save experiment config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)

    # Get configurations
    model_config = MODELS[experiment_config['model']]
    attention_config = ATTENTION[experiment_config['attention']]
    loss_config = LOSSES[experiment_config['loss']]
    dataset_config = DATASETS[experiment_config['dataset']]
    training_config = TRAINING[experiment_config['training']]
    feature_combo = FEATURE_COMBOS[experiment_config['feature_combo']]

    # Build datasets
    print("Building datasets...")
    train_dataset, val_dataset = build_datasets(dataset_config, feature_combo, is_classical=is_classical_model(model_config))
    experiment_config['normlisation_means'] = train_dataset.means.tolist()
    experiment_config['normlisation_stds'] = train_dataset.stds.tolist()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, train_dataset, device=device)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, val_dataset, device=device)
    )

    # Build model
    print("Building model...")
    num_features = len(feature_combo)
    num_bins = dataset_config['num_bins']
    model = build_model(
        model_config, attention_config, num_features, num_bins, device,
        feature_names=feature_combo
    )

    # Check if this is a classical ML model
    use_classical_training = is_classical_model(model_config)

    if use_classical_training:
        # Classical model training (uses fit() method)
        print("Training classical model...")
        history = model.fit(train_loader, device=device)
        history = to_json_safe(history)
        
        # Save training history
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Save classical model using pickle
        model.save(str(exp_dir / 'best_model.pkl'))

        # Evaluate classical model
        print("Evaluating model...")
        metrics = evaluate_classical_model(model, val_loader, device,
                                           visualise_predictions=10, save_dir=exp_dir / 'plots')

    else:
        # Deep learning model training (uses PyTorch training loop)
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])

        scheduler_type = training_config.get('scheduler', 'step')
        if scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=training_config.get('step_size', 10),
                gamma=training_config.get('gamma', 0.9)
            )
        else:
            scheduler = None

        history = None
        # Train model

        print("Training model...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=compute_loss,
            loss_config=loss_config,
            num_epochs=training_config['num_epochs'],
            device=device,
            save_path=exp_dir / 'best_model.pt',
            verbose=True
        )

        # Save training history
        with open(exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Plot training curves
        plot_losses(history, exp_dir / 'training_losses.png')

        # Load best model for evaluation
        model.load_state_dict(torch.load(exp_dir / 'best_model.pt')['model_state_dict'])

        # Evaluate on validation set
        print("Evaluating model...")
        metrics = evaluate_model(model, val_loader, device, visualise_predictions=10, save_dir=exp_dir / 'plots')

    # Save metrics
    with open(exp_dir / 'metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value
        json.dump(metrics_serializable, f, indent=2)

    optimal_accuracy = metrics['optimal_accuracy'] if 'optimal_accuracy' in metrics else None
    optimal_c_index = metrics['optimal_c_index'] if 'optimal_c_index' in metrics else None
    optimal_brier = metrics['optimal_brier_score'] if 'optimal_brier_score' in metrics else None
    
    # Print summary
    print(f"\nExperiment complete!")
    print(f"Results saved to: {exp_dir}")
    print(f"Validation C-index: {metrics['c_index']:.4f}", f'( / optimal: {optimal_c_index:.4f} )' if optimal_c_index is not None else '')
    print(f"Validation Brier Score: {metrics['brier_score']:.4f}", f'( / optimal: {optimal_brier:.4f} )' if optimal_brier is not None else '')
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}", f'( / optimal: {optimal_accuracy:.4f} )' if optimal_accuracy is not None else '')

    return {
        'name': experiment_config['name'],
        'metrics': metrics,
        'history': history,
        'config': experiment_config
    }



def run_experiment_suite(
    suite_name,
    output_dir,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    repeats: int = 1,
):
    """
    Run a predefined experiment suite.

    Args:
        suite_name: name of experiment suite from EXPERIMENT_SUITES
        output_dir: path to save results
        device: torch device
        repeats: number of times to repeat each experiment

    Returns:
        results: list of experiment results
    """
    if suite_name not in EXPERIMENT_SUITES:
        raise ValueError(f"Unknown experiment suite: {suite_name}")

    print(suite_name)
    experiments = EXPERIMENT_SUITES[suite_name] * repeats

    print(f"\n{'='*80}")
    print(f"Running experiment suite: {suite_name}")
    print(f"Total experiments: {len(experiments)}")
    print(f"{'='*80}\n")

    # if output_dir is None:
    #     # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / f"{suite_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)


    # Run all experiments
    results = []
    for i, exp_config in enumerate(experiments):
        print(f"\nExperiment {i+1}/{len(experiments)}")
        result = run_single_experiment(exp_config, output_dir, device)
        results.append(result)

    # Save summary
    summary = {
        'suite_name': suite_name,
        'timestamp': timestamp,
        'num_experiments': len(experiments),
        'results': [
            {
                'name': r['name'],
                'c_index': r['metrics']['c_index'] if 'metrics' in r else None,
                'brier_score': r['metrics']['brier_score'] if 'metrics' in r else None,
                'accuracy': r['metrics']['accuracy'] if 'metrics' in r else None,
                'error': r.get('error')
            }
            for r in results
        ]
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"Experiment Suite Complete: {suite_name}")
    print(f"{'='*80}")
    print(f"{'Experiment Name':<60} {'C-index':<10} {'Accuracy':<10}")
    print(f"{'-'*80}")
    for r in summary['results']:
        c_idx = f"{r['c_index']:.4f}" if r['c_index'] is not None else "ERROR"
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "ERROR"
        print(f"{r['name']:<60} {c_idx:<10} {acc:<10}")

    print(f"\nResults saved to: {output_dir}")
    
    plot_experiment_results(output_dir, plot_type='box')
    plot_confusion_matrices(output_dir)
    plot_experiment_losses(output_dir)

    return results

def evaluate_experiment(exp_dir: Path) -> dict:
    """
    Evaluate a single experiment from its directory.

    Args:
        experiment_dir: path to experiment directory
        
    Returns:

        results: dict with evaluation metrics#  """
    
    cfg = exp_dir / 'config.json'
    with open(cfg, 'r') as f:
        experiment_config = json.load(f)
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    model = build_model(
        MODELS[experiment_config['model']],
        ATTENTION[experiment_config['attention']],
        num_features=len(FEATURE_COMBOS[experiment_config['feature_combo']]),
        num_bins=DATASETS[experiment_config['dataset']]['num_bins'],
        device=device,
    )
    
    means, stds = experiment_config['normlisation_means'], experiment_config['normlisation_stds']
    
    val_dataset = CellDataset(
        hdf5_paths=DATASETS[experiment_config['dataset']]['val_paths'],
        features=FEATURE_COMBOS[experiment_config['feature_combo']],
        num_bins=DATASETS[experiment_config['dataset']]['num_bins'],
        min_length=DATASETS[experiment_config['dataset']]['min_length'],
        max_time_to_death=DATASETS[experiment_config['dataset']]['max_time_to_death'],
        means=means,
        stds=stds,
        preload_data=True,
    )
    
    # val_dataset = build_datasets(
    #     DATASETS[experiment_config['dataset']],
    #     FEATURE_COMBOS[experiment_config['feature_combo']]
    # )[1]
    
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=TRAINING[experiment_config['training']]['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, val_dataset, device=device),
    )
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(exp_dir / 'best_model.pt')['model_state_dict'])

    # Evaluate on validation set
    print("Evaluating model...")
    metrics = evaluate_model(model, val_loader, device, visualise_predictions=10, save_dir=exp_dir / 'plots')

    # Save metrics
    with open(exp_dir / 'metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = value
        json.dump(metrics_serializable, f, indent=2)
    
    return metrics


def evaluate_suite(suite_dir: Path) -> dict:
    """
    Evaluate all experiments in a directory and compile results.

    Args:
        suite_dir: path to experiment suite directory
    Returns:
        summary: dict with compiled results
    """
    experiment_dirs = [d for d in suite_dir.iterdir() if d.is_dir()]
    
    for i, exp_dir in enumerate(experiment_dirs):
        print(f"\nEvaluating expermient {i+1}/{len(experiment_dirs)}")
        evaluate_experiment(exp_dir)
        
    plot_experiment_results(suite_dir, plot_type='box')
    plot_confusion_matrices(suite_dir)
    plot_experiment_losses(suite_dir)

def interpret_model(model_path: Path) -> None:
    """
    Interpet a saved model.
    """
    cfg = model_path / 'config.json'
    with cfg.open('r') as f:
        config = json.load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = build_model(
        MODELS[config['model']],
        ATTENTION[config['attention']],
        num_features=len(FEATURE_COMBOS[config['feature_combo']]),
        num_bins=DATASETS[config['dataset']]['num_bins'],
        device=device,
    )
    model.load_state_dict(torch.load(model_path / 'best_model.pt')['model_state_dict'])
    
    test_dataset = CellDataset(
            hdf5_paths=DATASETS[config['dataset']]['train_paths'],
            features=FEATURE_COMBOS[config['feature_combo']],
            num_bins=DATASETS[config['dataset']]['num_bins'],
            min_length=DATASETS[config['dataset']]['min_length'],
            max_time_to_death=DATASETS[config['dataset']]['max_time_to_death'],
            preload_data=True,
        )
    
    background_data = create_background_from_dataset(
        dataset=test_dataset,
        num_samples=100,
        sample_type='sample',
        max_len=1000
    )
    
    # explainer = SurvivalSHAP(
    #     model=model,
    #     background_data=background_data,
    #     feature_names=FEATURE_COMBOS[config['feature_combo']],
    #     device=device,
    # )
    
    # results = explainer.analyse(test_dataset, method="gradient", num_time_bins=10, num_samples=500)
    # explainer.plot_summary(results, save_path=model_path / 'shap_summary.png')
         
    explainer = KernelSHAP(
        model=model,
        # background_data=background_data,
        feature_names=FEATURE_COMBOS[config['feature_combo']],
        device=device,
    )
    results = explainer.analyse_batch(test_dataset, num_segments=10, num_samples=500)
    explainer.plot_summary(results, save_path=model_path / 'kernel_shap_summary.png')

def main():
    parser = argparse.ArgumentParser(description='Run survival analysis experiments')
    parser.add_argument(
        '--suite',
        type=str,
        choices=list(EXPERIMENT_SUITES.keys()),
        help='Name of experiment suite to run'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='PhagoPred/survival_v2/experiments/results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--list-suites',
        action='store_true',
        help='List available experiment suites'
    )

    args = parser.parse_args()

    if args.list_suites:
        print("Available experiment suites:")
        for suite_name, experiments in EXPERIMENT_SUITES.items():
            print(f"  {suite_name}: {len(experiments)} experiments")
        return

    if args.suite is None:
        parser.print_help()
        print("\nPlease specify an experiment suite with --suite")
        return

    # Run experiment suite
    run_experiment_suite(
        suite_name=args.suite,
        output_dir=args.output,
        device=args.device
    )


if __name__ == '__main__':
    main()
