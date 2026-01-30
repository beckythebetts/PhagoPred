# Dataset Comparison Guide

This guide explains how to generate and compare survival models across different synthetic dataset variants.

## Overview

The framework now supports comprehensive dataset comparisons to test model robustness against:
- Different noise levels
- Late entry (left truncation)
- Missing data
- Different survival rules
- Feature subsets

## Step 1: Generate Dataset Variants

```bash
cd /home/ubuntu/PhagoPred
python PhagoPred/survival_v2/data/generate_datasets.py
```

This will create 14 dataset variants in `PhagoPred/Datasets/synthetic_variants/`:

### Generated Datasets

**Rule Variants:**
- `baseline_train.h5` / `baseline_val.h5` - All 3 rules (1400/600 cells)
- `single_rule_variation_train.h5` / `single_rule_variation_val.h5`
- `single_rule_ramp_train.h5` / `single_rule_ramp_val.h5`
- `no_rules_train.h5` / `no_rules_val.h5`

**Noise Variants:**
- `low_noise_train.h5` / `low_noise_val.h5` (σ=0.001)
- `medium_noise_train.h5` / `medium_noise_val.h5` (σ=0.1)
- `high_noise_train.h5` / `high_noise_val.h5` (σ=0.5)

**Late Entry Variants:**
- `late_entry_30pct_train.h5` / `late_entry_30pct_val.h5`
- `late_entry_50pct_train.h5` / `late_entry_50pct_val.h5`
- `late_entry_extreme_train.h5` / `late_entry_extreme_val.h5`

**Missing Data Variants:**
- `missing_data_10pct_train.h5` / `missing_data_10pct_val.h5`
- `missing_data_20pct_train.h5` / `missing_data_20pct_val.h5`

**Combined Challenges:**
- `challenging_train.h5` / `challenging_val.h5` (high noise + 30% late entry + 10% missing)

## Step 2: Run Comparison Experiments

### Quick Test on Noise Levels

```bash
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite noise_comparison \
    --output PhagoPred/survival_v2/experiments/results \
    --device cuda
```

This compares LSTM performance on:
- Low noise (σ=0.001)
- Baseline (σ=0.01)
- Medium noise (σ=0.1)
- High noise (σ=0.5)

### Late Entry Comparison

```bash
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite late_entry_comparison \
    --output PhagoPred/survival_v2/experiments/results \
    --device cuda
```

Compares performance with:
- Baseline (no late entry)
- 30% late entry
- 50% late entry
- 70% late entry (extreme)

### Full Robustness Test

```bash
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite dataset_robustness \
    --output PhagoPred/survival_v2/experiments/results \
    --device cuda
```

Tests both LSTM and CNN on:
- Baseline
- High noise
- 50% late entry
- 10% missing data
- Combined challenges

### Complete Dataset Sweep

⚠️ **Warning**: This runs 15 experiments (one per dataset)

```bash
python PhagoPred/survival_v2/experiments/run_experiments.py \
    --suite full_dataset_sweep \
    --output PhagoPred/survival_v2/experiments/results \
    --device cuda
```

## Step 3: Analyze Results

After experiments complete, results are saved in timestamped directories:

```
results/
└── noise_comparison_20260108_143025/
    ├── summary.json                    # Overall comparison
    ├── lstm_medium_vector_nll_only_low_noise_standard/
    │   ├── config.json
    │   ├── best_model.pt
    │   ├── training_history.json
    │   ├── training_curves.png
    │   └── metrics.json
    ├── lstm_medium_vector_nll_only_baseline_standard/
    │   └── ...
    └── ...
```

### Summary Table

The `summary.json` file contains a comparison table. You can also see it printed at the end of training:

```
Experiment Name                                              C-index    Accuracy  
--------------------------------------------------------------------------------
lstm_medium_vector_nll_only_low_noise_standard              0.7234     0.6521    
lstm_medium_vector_nll_only_baseline_standard               0.7145     0.6432    
lstm_medium_vector_nll_only_medium_noise_standard           0.6892     0.6123    
lstm_medium_vector_nll_only_high_noise_standard             0.6234     0.5645    
```

## Available Experiment Suites

| Suite | Description | # Experiments |
|-------|-------------|---------------|
| `noise_comparison` | 4 noise levels | 4 |
| `late_entry_comparison` | 4 late entry proportions | 4 |
| `missing_data_comparison` | 3 missing data levels | 3 |
| `rule_comparison` | 4 rule configurations | 4 |
| `feature_comparison` | 3 feature subsets | 3 |
| `dataset_robustness` | 5 datasets × 2 models | 10 |
| `full_dataset_sweep` | All 15 datasets | 15 |

## Custom Experiments

### Test Specific Dataset with Different Models

```python
from PhagoPred.survival_v2.configs.experiment_config import generate_experiment_grid
from PhagoPred.survival_v2.experiments.run_experiments import run_experiment_suite

experiments = generate_experiment_grid(
    models=['lstm_small', 'lstm_medium', 'lstm_large', 'cnn_medium'],
    attentions=['vector'],
    losses=['nll_only'],
    datasets=['challenging'],  # Most difficult dataset
    trainings=['standard']
)

# Run manually or save to custom suite
```

### Compare Attention Mechanisms on Challenging Dataset

```python
experiments = generate_experiment_grid(
    models=['lstm_medium'],
    attentions=['vector', 'fc_medium', 'multihead_4'],
    losses=['nll_only'],
    datasets=['challenging', 'baseline'],
    trainings=['standard']
)
```

## Interpreting Results

### C-index (Concordance Index)
- Measures ranking quality: higher is better
- **0.5**: Random performance
- **0.7+**: Good performance
- **0.8+**: Excellent performance

### Accuracy
- Correct time bin classification
- Depends on number of bins (5 bins → 20% random baseline)

### Expected Performance Trends

1. **Noise Impact**: C-index should decrease as noise increases
2. **Late Entry**: Should handle well if model correctly handles left truncation
3. **Missing Data**: Robustness depends on how NaNs are handled
4. **Rules**: More rules → stronger survival signal → better performance
5. **Features**: More relevant features → better performance

## Tips

1. **Start Small**: Run `quick_test` suite first to verify setup
2. **Use GPU**: Add `--device cuda` for faster training
3. **Monitor Training**: Check `training_curves.png` for overfitting
4. **Compare Fairly**: Always use same training config when comparing datasets
5. **Reproducibility**: Dataset generation uses fixed seeds for reproducibility

## Troubleshooting

**Out of Memory?**
- Reduce batch size in training configs
- Use smaller models (lstm_small, cnn_small)

**Datasets not found?**
- Run `generate_datasets.py` first
- Check paths in `experiment_config.py`

**Poor Performance?**
- Check `training_curves.png` for underfitting/overfitting
- Try different learning rates or longer training
- Verify dataset statistics (noise level, late entry %)

