"""
Experiment configuration templates.
"""

# Model architectures
MODELS = {
    'lstm_small': {
        'type': 'lstm',
        'lstm_hidden_size': 32,
        'lstm_num_layers': 1,
        'lstm_dropout': 0.0,
        'fc_layers': [32],
        'use_predictor': False
    },
    'lstm_medium': {
        'type': 'lstm',
        'lstm_hidden_size': 64,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.2,
        'fc_layers': [64, 32],
        'use_predictor': False
    },
    'lstm_large': {
        'type': 'lstm',
        'lstm_hidden_size': 128,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.2,
        'predictor_layers': [64],
        'fc_layers': [128, 64],
        'use_predictor': True
    },
    'cnn_small': {
        'type': 'cnn',
        'num_channels': [32, 32, 32],
        'kernel_sizes': [3, 3, 3],
        'dilations': [1, 2, 4],
        'fc_layers': [32]
    },
    'cnn_medium': {
        'type': 'cnn',
        'num_channels': [64] * 8,
        'kernel_sizes': [3] * 8,
        'dilations': [2**n for n in range(8)],
        'fc_layers': [64, 32]
    },
    'cnn_large': {
        'type': 'cnn',
        'num_channels': [128] * 10,
        'kernel_sizes': [3] * 10,
        'dilations': [2**n for n in range(10)],
        'fc_layers': [128, 64, 32]
    },

    # ========== Classical ML Models ==========

    # Random Forest variants
    'rf_summary': {
        'type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'feature_extraction': 'summary'
    },
    'rf_segments': {
        'type': 'random_forest',
        'n_estimators': 100,
        'max_depth': 10,
        'feature_extraction': 'segments',
        'n_segments': 10
    },
    'rf_temporal': {
        'type': 'random_forest',
        'n_estimators': 200,
        'max_depth': 15,
        'feature_extraction': 'temporal_full',
        'n_segments': 5,
        'n_lags': 10
    },

    # XGBoost variants
    'xgb_summary': {
        'type': 'xgboost',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'feature_extraction': 'summary'
    },
    'xgb_temporal': {
        'type': 'xgboost',
        'n_estimators': 200,
        'max_depth': 8,
        'learning_rate': 0.1,
        'feature_extraction': 'temporal_full',
        'n_segments': 5,
        'n_lags': 10
    },

    # LightGBM variants
    'lgbm_summary': {
        'type': 'lightgbm',
        'n_estimators': 100,
        'max_depth': -1,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_extraction': 'summary'
    },
    'lgbm_temporal': {
        'type': 'lightgbm',
        'n_estimators': 200,
        'max_depth': -1,
        'num_leaves': 63,
        'learning_rate': 0.1,
        'feature_extraction': 'temporal_full',
        'n_segments': 5,
        'n_lags': 10
    },

    # Scikit-survival models (proper survival analysis)
    'rsf': {
        'type': 'random_survival_forest',
        'n_estimators': 100,
        'max_depth': None,
        'feature_extraction': 'temporal_full',
        'n_segments': 5,
        'n_lags': 10
    },
    'gbsa': {
        'type': 'gradient_boosting_survival',
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'feature_extraction': 'temporal_full',
        'n_segments': 5,
        'n_lags': 10
    },
}

# Attention mechanisms
ATTENTION = {
    'none_mean': {
        'type': 'mean_pool'
    },
    'none_last': {
        'type': 'last_pool'
    },
    'vector': {
        'type': 'vector'
    },
    'fc_small': {
        'type': 'fc',
        'hidden_dims': [32]
    },
    'fc_medium': {
        'type': 'fc',
        'hidden_dims': [64, 32]
    },
    'multihead_2': {
        'type': 'multihead',
        'num_heads': 2,
        'dropout': 0.1
    },
    # 'multihead_4': {
    #     'type': 'multihead',
    #     'num_heads': 4,
    #     'dropout': 0.1
    # },
}

# Loss configurations
LOSSES = {
    'nll_only': {
        'nll': 1.0,
        'nll_type': 'standard',
        'ranking': 0.0,
        'prediction': 0.0
    },
    'soft_target': {
        'nll': 1.0,
        'nll_type': 'soft_target',
        'soft_target_sigma': 0.8,
        'ranking': 0.0,
        'prediction': 0.0
    },
    'soft_target_tight': {
        'nll': 1.0,
        'nll_type': 'soft_target',
        'soft_target_sigma': 0.5,
        'ranking': 0.0,
        'prediction': 0.0
    },
    'soft_target_wide': {
        'nll': 1.0,
        'nll_type': 'soft_target',
        'soft_target_sigma': 1.0,
        'ranking': 0.0,
        'prediction': 0.0
    },
    'nll_ranking_weak': {
        'nll': 1.0,
        'nll_type': 'standard',
        'ranking': 0.05,
        'ranking_type': 'concordance',
        'prediction': 0.0
    },
    'nll_ranking_strong': {
        'nll': 1.0,
        'nll_type': 'standard',
        'ranking': 0.2,
        'ranking_type': 'concordance',
        'prediction': 0.0
    },
    'soft_target_ranking': {
        'nll': 1.0,
        'nll_type': 'soft_target',
        'soft_target_sigma': 0.8,
        'ranking': 0.05,
        'ranking_type': 'concordance',
        'prediction': 0.0
    },
    'nll_prediction': {
        'nll': 1.0,
        'nll_type': 'standard',
        'ranking': 0.0,
        'prediction': 0.5
    },
    'full': {
        'nll': 1.0,
        'nll_type': 'standard',
        'ranking': 0.1,
        'ranking_type': 'concordance',
        'prediction': 0.5
    },
}

# Dataset configurations
DATASETS = {
    # Original synthetic dataset
    'synthetic_basic': {
        'train_paths': ['PhagoPred/Datasets/synthetic.h5'],
        'val_paths': ['PhagoPred/Datasets/val_synthetic.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },

    # Baseline - clean dataset with all rules
    'baseline': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 50,
        'max_time_to_death': 100
    },

    # Single rule variants
    'single_rule_variation': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/single_rule_variation_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/single_rule_variation_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'single_rule_ramp': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/single_rule_ramp_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/single_rule_ramp_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'no_rules': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/no_rules_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/no_rules_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },

    # Different noise levels
    'low_noise': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/low_noise_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/low_noise_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'medium_noise': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/medium_noise_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/medium_noise_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'high_noise': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/high_noise_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/high_noise_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },

    # Late entry variants
    'late_entry_30pct': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_30pct_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_30pct_val.h5'],
        # 'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 50,
        'max_time_to_death': 100
    },
    'late_entry_50pct': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_50pct_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_50pct_val.h5'],
        # 'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 50,
        'max_time_to_death': 100
    },
    'late_entry_extreme': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_val.h5'],
        # 'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 50,
        'max_time_to_death': 100
    },
    'late_entry_extreme_start_feature': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/late_entry_extreme_val.h5'],
        # 'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 50,
        'max_time_to_death': 100,
        'add_start_frame_feature': True,
    },
    
    # Missing data variants
    'missing_data_10pct': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/missing_data_10pct_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/missing_data_10pct_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'missing_data_20pct': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/missing_data_20pct_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/missing_data_20pct_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },

    # Combined challenges
    'challenging': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/challenging_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/challenging_val.h5'],
        'features': ['0', '1', '2', '3'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },

    # Feature subset experiments (using baseline dataset)
    'features_012': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        'features': ['0', '1', '2'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
    'features_01': {
        'train_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_train.h5'],
        'val_paths': ['PhagoPred/Datasets/synthetic_variants/baseline_val.h5'],
        'features': ['0', '1'],
        'num_bins': 5,
        'min_length': 500,
        'max_time_to_death': 100
    },
}

FEATURE_COMBOS = {
    'all': ['0', '1', '2', '3'],
    'no_frame_count': ['0', '1', '3'],
}


# Training configurations
TRAINING = {
    'quick_test': {
        'num_epochs': 10,
        'batch_size': 128,
        'lr': 1e-3,
        'scheduler': 'step',
        'step_size': 5,
        'gamma': 0.9
    },
    'standard': {
        'num_epochs': 150,
        'batch_size': 256,
        'lr': 1e-3,
        'scheduler': 'step',
        'step_size': 10,
        'gamma': 0.9
    },
    'long': {
        'num_epochs': 250,
        'batch_size': 256,
        'lr': 1e-3,
        'scheduler': 'step',
        'step_size': 15,
        'gamma': 0.9
    },
}


def generate_experiment_grid(
    models=None,
    attentions=None,
    losses=None,
    datasets=None,
    trainings=None,
    feature_combos=None,
):
    """
    Generate all combinations of experiments.

    Args:
        models: list of model config names (or None for all)
        attentions: list of attention config names (or None for all)
        losses: list of loss config names (or None for all)
        datasets: list of dataset config names (or None for all)
        trainings: list of training config names (or None for all)

    Returns:
        list of experiment dicts
    """
    # print(models, feature_combos)
    if models is None:
        models = list(MODELS.keys())
    if attentions is None:
        attentions = list(ATTENTION.keys())
    if losses is None:
        losses = list(LOSSES.keys())
    if datasets is None:
        datasets = list(DATASETS.keys())
    if trainings is None:
        trainings = list(TRAINING.keys())
    if feature_combos is None:
        feature_combos = list(FEATURE_COMBOS.keys())[0:1]    
    

    experiments = []
    for model_name in models:
        for attn_name in attentions:
            for loss_name in losses:
                for dataset_name in datasets:
                    for training_name in trainings:
                        for feature_combo in feature_combos:
                            exp = {
                                'name': f"{model_name}_{attn_name}_{loss_name}_{dataset_name}_{training_name}_{feature_combo}",
                                'model': model_name,
                                'attention': attn_name,
                                'loss': loss_name,
                                'dataset': dataset_name,
                                'training': training_name,
                                'feature_combo': feature_combo,
                            }
                            experiments.append(exp)

    return experiments


# Predefined experiment suites
EXPERIMENT_SUITES = {
    'attention_comparison': generate_experiment_grid(
        models=['cnn_medium', 'lstm_medium'],
        attentions=list(ATTENTION.keys()),
        losses=['soft_target'],
        datasets=['late_entry_extreme'],
        trainings=['standard']
    ),
    'loss_comparison': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=list(LOSSES.keys()),
        datasets=['synthetic_basic'],
        trainings=['standard']
    ),
    'model_comparison': generate_experiment_grid(
        models=list(MODELS.keys()),
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['synthetic_basic'],
        trainings=['standard']
    ),
    'quick_test': generate_experiment_grid(
        models=['lstm_small', 'cnn_small'],
        attentions=['vector', 'none_mean'],
        losses=['nll_only'],
        datasets=['synthetic_basic'],
        trainings=['quick_test']
    ),

    # Soft target loss comparison
    'soft_target_comparison': generate_experiment_grid(
        models=['lstm_medium', 'cnn_medium'],
        attentions=['vector'],
        losses=['nll_only', 'soft_target', 'soft_target_tight', 'soft_target_wide'],
        datasets=['baseline'],
        trainings=['standard']
    ),
    'soft_target_sigma_sweep': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['soft_target_tight', 'soft_target', 'soft_target_wide'],
        datasets=['baseline'],
        trainings=['standard']
    ),
    'soft_target_with_ranking': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only', 'soft_target', 'nll_ranking_weak', 'soft_target_ranking'],
        datasets=['baseline'],
        trainings=['standard']
    ),

    # Dataset comparison suites
    'noise_comparison': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['low_noise', 'baseline', 'medium_noise', 'high_noise'],
        trainings=['standard']
    ),
    'late_entry_comparison': generate_experiment_grid(
        models=['lstm_medium', 'cnn_medium'],
        attentions=['none_last'],
        losses=['soft_target'],
        datasets=['baseline', 'late_entry_30pct', 'late_entry_50pct', 'late_entry_extreme'],
        trainings=['standard']
    ),
    'missing_data_comparison': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['baseline', 'missing_data_10pct', 'missing_data_20pct'],
        trainings=['standard']
    ),
    'rule_comparison': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['no_rules', 'single_rule_variation', 'single_rule_ramp', 'baseline'],
        trainings=['standard']
    ),
    'feature_comparison': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['features_01', 'features_012', 'baseline'],
        trainings=['standard']
    ),
    
    'framecount_feature_comparison': generate_experiment_grid(
        models=['cnn_medium'],
        # attentions=['none_last'],
        attentions = list(ATTENTION.keys()),
        losses=['soft_target'],
        datasets=['late_entry_extreme'],
        trainings=['standard'],
        feature_combos=['all', 'no_frame_count'],
    ),
    
    'start_frame_feature_comparison': generate_experiment_grid(
        models=['cnn_medium'],
        attentions=['none_last'],
        losses=['soft_target'],
        datasets=['late_entry_extreme', 'late_entry_extreme_start_feature'],
        trainings=['standard'],
        feature_combos=['no_frame_count'],
    ),

    # Comprehensive dataset comparison with multiple models
    'dataset_robustness': generate_experiment_grid(
        models=['lstm_medium', 'cnn_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=['baseline', 'high_noise', 'late_entry_50pct', 'missing_data_10pct', 'challenging'],
        trainings=['standard']
    ),

    # Full sweep across all dataset variants (use with caution - many experiments!)
    'full_dataset_sweep': generate_experiment_grid(
        models=['lstm_medium'],
        attentions=['vector'],
        losses=['nll_only'],
        datasets=list(DATASETS.keys()),
        trainings=['standard']
    ),

    # ========== Classical ML Experiment Suites ==========

    # Quick test for classical models
    'classical_quick_test': generate_experiment_grid(
        models=['rf_summary', 'xgb_summary'],
        attentions=['none_mean'],  # Not used but required by config
        losses=['nll_only'],
        datasets=['baseline'],
        trainings=['quick_test']
    ),

    # Feature extraction method comparison
    'feature_extraction_comparison': generate_experiment_grid(
        models=['rf_summary', 'rf_segments', 'rf_temporal'],
        attentions=['none_mean'],
        losses=['nll_only'],
        datasets=['baseline'],
        trainings=['standard']
    ),

    # Classical vs Deep Learning comparison
    'classical_vs_dl': generate_experiment_grid(
        models=['rf_temporal', 'xgb_temporal', 'lstm_medium', 'cnn_medium'],
        attentions=['vector'],
        losses=['soft_target'],
        datasets=['baseline', 'late_entry_50pct'],
        trainings=['standard']
    ),

    # Full classical model comparison
    'classical_comparison': generate_experiment_grid(
        models=['rf_summary', 'rf_temporal', 'xgb_summary', 'xgb_temporal'],
        attentions=['none_mean'],
        losses=['nll_only'],
        datasets=['baseline', 'late_entry_extreme'],
        trainings=['standard']
    ),

    # Classical models on different datasets
    'classical_dataset_robustness': generate_experiment_grid(
        models=['rf_temporal', 'xgb_temporal'],
        attentions=['none_mean'],
        losses=['nll_only'],
        datasets=['baseline', 'high_noise', 'late_entry_50pct', 'missing_data_10pct'],
        trainings=['standard']
    ),
}
