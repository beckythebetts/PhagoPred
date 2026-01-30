"""
Generate multiple synthetic datasets with different configurations.
"""
from pathlib import Path
from PhagoPred.survival_v2.data.synthetic_data import (
    create_synthetic_dataset,
    VariationRule,
    GradualRampRule
)

# Output directory
output_dir = Path('PhagoPred/Datasets/synthetic_variants')
output_dir.mkdir(parents=True, exist_ok=True)

# Dataset configurations
dataset_configs = {
    # Baseline dataset (default settings)
    'baseline': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'late_entry_prob': 0.0,
        'feature_mask_prob': 0.0,
        'seed': 42,
    },

    # Single rule variants
    'single_rule_variation': {
        'num_cells': 2000,
        'num_frames': 1000,
        'rules': [VariationRule(feature='3', delay=150, sigma=20.0)],
        'noise_level': 0.01,
        'seed': 42,
    },
    'single_rule_ramp': {
        'num_cells': 2000,
        'num_frames': 1000,
        'rules': [GradualRampRule(feature='0', ramp_height=10.0, delay=200, sigma=30.0)],
        'noise_level': 0.01,
        'seed': 42,
    },

    # No rules (pure noise)
    'no_rules': {
        'num_cells': 2000,
        'num_frames': 1000,
        'rules': [],
        'noise_level': 0.01,
        'seed': 42,
    },

    # Different noise levels
    'low_noise': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.001,
        'seed': 42,
    },
    'medium_noise': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.1,
        'seed': 42,
    },
    'high_noise': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.5,
        'seed': 42,
    },

    # Late entry variants
    'late_entry_30pct': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'late_entry_prob': 0.3,
        'late_entry_range': (100, 300),
        'seed': 42,
    },
    'late_entry_50pct': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'late_entry_prob': 0.5,
        'late_entry_range': (100, 300),
        'seed': 42,
    },
    'late_entry_extreme': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'late_entry_prob': 0.7,
        'late_entry_range': (200, 500),
        'seed': 42,
    },

    # Feature masking (missing data)
    'missing_data_10pct': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'feature_mask_prob': 0.1,
        'seed': 42,
    },
    'missing_data_20pct': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.01,
        'feature_mask_prob': 0.2,
        'seed': 42,
    },

    # Combined challenges
    'challenging': {
        'num_cells': 2000,
        'num_frames': 1000,
        'noise_level': 0.1,
        'late_entry_prob': 0.3,
        'late_entry_range': (100, 300),
        'feature_mask_prob': 0.1,
        'seed': 42,
    },
}


def generate_all_datasets(train_val_split=0.7):
    """Generate all dataset variants with train/val splits."""
    for name, config in dataset_configs.items():
        print(f"\n{'='*80}")
        print(f"Generating dataset: {name}")
        print(f"{'='*80}")

        # Training set
        train_file = output_dir / f'{name}_train.h5'
        
        # if train_file.exists():
        #     print(f"Dataset already exists: {train_file}, skipping.")
        #     return
            
        print(f"\nCreating training set: {train_file}")
        train_config = config.copy()
        train_config['num_cells'] = int(config['num_cells'] * train_val_split)
        create_synthetic_dataset(filename=train_file, **train_config)

        # Validation set (different seed for different samples)
        val_file = output_dir / f'{name}_val.h5'

        print(f"\nCreating validation set: {val_file}")
        val_config = config.copy()
        val_config['num_cells'] = int(config['num_cells'] * (1 - train_val_split))
        val_config['seed'] = config.get('seed', 42) + 1000 if config.get('seed') is not None else None
        create_synthetic_dataset(filename=val_file, **val_config)

    print(f"\n{'='*80}")
    print(f"All datasets generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    generate_all_datasets()
