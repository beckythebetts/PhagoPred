# Survival Analysis Framework v2

A modular framework for survival analysis with deep learning models, featuring pluggable attention mechanisms and flexible loss configurations.

## Features

- **Modular Architecture**: Shared components, no code duplication
- **Pluggable Attention**: Choose from 5 different attention mechanisms
- **Flexible Loss Functions**: Mix and match NLL, ranking, and prediction losses
- **Multiple Models**: LSTM and CNN-based survival models
- **Experiment Framework**: Run and compare multiple configurations automatically

## Directory Structure

```
survival_v2/
├── attention/          # Attention mechanisms
│   ├── mechanisms.py   # 5 attention types (vector, fc, multihead, mean, last)
│   └── __init__.py
├── configs/            # Experiment configurations
│   ├── experiment_config.py
│   └── __init__.py
├── data/               # Dataset utilities
│   ├── dataset.py      # CellDataset class
│   ├── synthetic_data.py
│   └── __init__.py
├── experiments/        # Experiment runner
│   ├── run_experiments.py
│   └── __init__.py
├── losses/             # Loss functions
│   ├── survival_losses.py
│   └── __init__.py
├── models/             # Survival models
│   ├── base.py         # Base SurvivalModel class
│   ├── lstm_survival.py
│   ├── cnn_survival.py
│   └── __init__.py
├── utils/              # Training utilities
│   ├── training.py     # train_epoch, validate_epoch, train_model
│   ├── metrics.py      # C-index, calibration, etc.
│   └── __init__.py
└── README.md
```

## Quick Start

### 1. Running Predefined Experiment Suites

```bash
# List available experiment suites
python PhagoPred/survival_v2/experiments/run_experiments.py --list-suites

# Run an experiment suite
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite attention_comparison \
    --output PhagoPred/survival_v2/experiments/results
```

Available suites:
- `attention_comparison`: Compare all attention mechanisms (LSTM & CNN with different attention)
- `loss_comparison`: Compare different loss combinations
- `model_comparison`: Compare all model architectures
- `quick_test`: Fast test with small models
- **`noise_comparison`**: Compare performance across different noise levels
- **`late_entry_comparison`**: Compare with different late entry proportions
- **`missing_data_comparison`**: Test robustness to missing data
- **`rule_comparison`**: Compare single rules vs multiple rules
- **`feature_comparison`**: Test with different feature subsets
- **`dataset_robustness`**: Comprehensive test across challenging datasets
- **`full_dataset_sweep`**: Complete sweep across all dataset variants

### 2. Custom Experiments

```python
from PhagoPred.survival_v2.configs.experiment_config import generate_experiment_grid

# Generate custom experiment grid
experiments = generate_experiment_grid(
    models=['lstm_medium', 'cnn_medium'],
    attentions=['vector', 'multihead_4'],
    losses=['nll_only', 'nll_ranking_weak'],
    datasets=['synthetic_basic'],
    trainings=['standard']
)

# Run experiments (see run_experiments.py for full example)
```

### 3. Training a Single Model

```python
import torch
from torch.utils.data import DataLoader
from PhagoPred.survival_v2.models import LSTMSurvival
from PhagoPred.survival_v2.losses import compute_loss
from PhagoPred.survival_v2.utils import train_model
from PhagoPred.survival_v2.data import CellDataset, collate_fn

# Load dataset
train_dataset = CellDataset(
    hdf5_paths=['PhagoPred/Datasets/synthetic.h5'],
    features=['0', '1', '2', '3'],
    num_bins=5,
    min_length=500,
    max_time_to_death=100,
    preload_data=True
)

# Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, train_dataset, device='cuda')
)

# Build model with multihead attention
model = LSTMSurvival(
    num_features=4,
    num_bins=5,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_dropout=0.2,
    fc_layers=[64, 32],
    attention_type='multihead',  # or 'vector', 'fc', 'mean_pool', 'last_pool'
    attention_config={'num_heads': 4, 'dropout': 0.1}
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

# Loss configuration
loss_config = {
    'nll': 1.0,
    'ranking': 0.1,
    'ranking_type': 'concordance',
    'prediction': 0.0
}

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=compute_loss,
    loss_config=loss_config,
    num_epochs=50,
    device='cuda',
    save_path='best_model.pt'
)
```

## Attention Mechanisms

Choose the right attention mechanism for your task:

| Mechanism | Parameters | Use Case |
|-----------|-----------|----------|
| `vector` | Minimal (embed_dim) | Parameter-efficient, good baseline |
| `fc` | Low-Medium | More expressive than vector, still efficient |
| `multihead` | High (4×embed_dim²) | Most expressive, may overfit with limited data |
| `mean_pool` | Zero | No learnable parameters, simple average |
| `last_pool` | Zero | Use last valid timestep (good for LSTMs) |

Example configurations:

```python
# Simple attention vector
attention_config = {'type': 'vector'}

# FC attention with small MLP
attention_config = {
    'type': 'fc',
    'hidden_dims': [64, 32],
    'dropout': 0.1
}

# Multi-head self-attention
attention_config = {
    'type': 'multihead',
    'num_heads': 4,
    'dropout': 0.1
}
```

## Loss Functions

Configure loss combinations to suit your problem:

### Predefined Configurations

```python
LOSS_CONFIGS = {
    'nll_only': {
        'nll': 1.0,
        'ranking': 0.0,
        'prediction': 0.0
    },
    'nll_ranking_weak': {
        'nll': 1.0,
        'ranking': 0.05,
        'ranking_type': 'concordance',
        'prediction': 0.0
    },
    'nll_ranking_strong': {
        'nll': 1.0,
        'ranking': 0.2,
        'ranking_type': 'concordance',
        'prediction': 0.0
    },
    'nll_prediction': {
        'nll': 1.0,
        'ranking': 0.0,
        'prediction': 0.5  # For LSTMs with feature predictor
    },
    'full': {
        'nll': 1.0,
        'ranking': 0.1,
        'ranking_type': 'concordance',
        'prediction': 0.5
    }
}
```

### Loss Components

- **NLL (Negative Log-Likelihood)**: Primary loss for survival prediction
- **Ranking Loss**: Concordance-based ranking (use `'concordance'`, NOT `'cif'`)
- **Prediction Loss**: MSE for LSTM feature prediction (requires `use_predictor=True`)

**Important**: Use `ranking_type='concordance'` for single-cause problems. The CIF-based ranking loss pushes probability mass to the final time bin and is only appropriate for competing risks.

## Models

### LSTM Survival Model

Bidirectional LSTM with optional feature predictor:

```python
model = LSTMSurvival(
    num_features=4,
    num_bins=5,
    lstm_hidden_size=64,
    lstm_num_layers=2,
    lstm_dropout=0.2,
    fc_layers=[64, 32],
    attention_type='multihead',
    attention_config={'num_heads': 4},
    use_predictor=True,  # Enable feature prediction
    predictor_layers=[64]
)
```

### CNN Survival Model

Temporal CNN with dilated convolutions:

```python
model = CNNSurvival(
    num_features=4,
    num_bins=5,
    num_channels=[64, 64, 64],
    kernel_sizes=[3, 3, 3],
    dilations=[1, 2, 4],
    fc_layers=[64, 32],
    attention_type='vector',
    attention_config={}
)
```

## Configuration Templates

All configurations are defined in [configs/experiment_config.py](configs/experiment_config.py):

- **MODELS**: Architecture configurations (lstm_small/medium/large, cnn_small/medium/large)
- **ATTENTION**: Attention mechanism configurations
- **LOSSES**: Loss weight configurations
- **DATASETS**: Dataset paths and parameters
- **TRAINING**: Training hyperparameters (epochs, batch size, learning rate, scheduler)

## Output Format

Each experiment produces:

```
results/
├── experiment_name/
│   ├── config.json              # Experiment configuration
│   ├── best_model.pt            # Best model checkpoint
│   ├── training_history.json    # Loss history per epoch
│   ├── training_curves.png      # Train/val loss plots
│   └── metrics.json             # Final metrics (C-index, accuracy, etc.)
```

## Evaluation Metrics

Models are evaluated using:

- **C-index (Concordance Index)**: Ranking metric for survival predictions
- **Accuracy**: Correct time bin classification
- **Calibration Error**: Mean absolute difference between predicted and observed probabilities
- **Confusion Matrix**: Predicted vs actual time bins

## Tips

1. **Start with NLL-only loss**: The ranking loss can be problematic for single-cause problems
2. **Use concordance-based ranking**: If you need ranking loss, use `ranking_type='concordance'`
3. **Choose attention wisely**: Start with `vector` or `fc` before trying `multihead`
4. **Monitor overfitting**: Use validation loss for early stopping
5. **Normalize features**: The dataset automatically computes normalization stats

## Example Workflow

```bash
# 1. Run quick test to verify setup
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite quick_test \
    --output results/

# 2. Compare attention mechanisms
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite attention_comparison \
    --output results/

# 3. Compare loss configurations
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite loss_comparison \
    --output results/

# 4. Full model comparison
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite model_comparison \
    --output results/
```

Results will include summary tables comparing all experiments.

## Generating Synthetic Datasets

The framework supports generating multiple synthetic dataset variants to test model robustness.

### Quick Dataset Generation

```bash
# Generate all dataset variants
python PhagoPred/survival_v2/data/generate_datasets.py
```

This creates datasets with different characteristics:

### Dataset Variants

| Category | Dataset | Description |
|----------|---------|-------------|
| **Rules** | `baseline` | All 3 rules active (variation, 2x ramps) |
| | `single_rule_variation` | Only variation rule |
| | `single_rule_ramp` | Only ramp rule |
| | `no_rules` | Pure noise, no survival rules |
| **Noise** | `low_noise` | σ=0.001 (very clean) |
| | `baseline` | σ=0.01 (standard) |
| | `medium_noise` | σ=0.1 (noisy) |
| | `high_noise` | σ=0.5 (very noisy) |
| **Late Entry** | `late_entry_30pct` | 30% late entry (frames 100-300) |
| | `late_entry_50pct` | 50% late entry (frames 100-300) |
| | `late_entry_extreme` | 70% late entry (frames 200-500) |
| **Missing Data** | `missing_data_10pct` | 10% feature values randomly masked |
| | `missing_data_20pct` | 20% feature values randomly masked |
| **Combined** | `challenging` | High noise + 30% late entry + 10% missing |
| **Features** | `features_012` | Only features 0, 1, 2 |
| | `features_01` | Only features 0, 1 |

### Custom Dataset Generation

```python
from pathlib import Path
from PhagoPred.survival_v2.data.synthetic_data import (
    create_synthetic_dataset,
    VariationRule,
    GradualRampRule
)

# Create custom dataset
create_synthetic_dataset(
    filename=Path('custom_dataset.h5'),
    num_cells=2000,
    num_frames=1000,
    rules=[
        VariationRule(feature='3', delay=150, sigma=20.0),
        GradualRampRule(feature='0', ramp_height=10.0, delay=200, sigma=30.0)
    ],
    noise_level=0.05,           # Medium noise
    late_entry_prob=0.3,        # 30% late entry
    late_entry_range=(100, 300), # Late entry between frames 100-300
    feature_mask_prob=0.1,      # 10% missing data
    seed=42                     # Reproducibility
)
```

### Dataset Configuration Parameters

- **`rules`**: List of Rule objects (VariationRule, GradualRampRule) that define survival patterns
- **`noise_level`**: Standard deviation of Gaussian noise added to features
- **`late_entry_prob`**: Fraction of samples with late entry (0 to 1)
- **`late_entry_range`**: (min, max) frames for late entry start times
- **`feature_mask_prob`**: Fraction of feature values to randomly mask with NaN
- **`seed`**: Random seed for reproducibility

### Running Dataset Comparison Experiments

```bash
# Compare performance across noise levels
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite noise_comparison \
    --output results/

# Test late entry robustness
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite late_entry_comparison \
    --output results/

# Comprehensive robustness test
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite dataset_robustness \
    --output results/
```

Results will include C-index and accuracy for each dataset variant, allowing you to assess model robustness.

