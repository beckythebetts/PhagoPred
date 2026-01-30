"""Diagnostic script to understand how different delay values affect the synthetic dataset."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import h5py

from PhagoPred.survival_analysis.data.synthetic_data import Cell, GradualRampRule, create_synthetic_dataset
from PhagoPred.survival_analysis.data.dataset import CellDataset

def create_dataset_with_delay(delay, num_cells=1000, num_frames=1000):
    """Create synthetic dataset with specified delay."""

    # Create temporary file
    temp_path = Path('temp') / f'synthetic_delay_{delay}.h5'
    temp_path.parent.mkdir(exist_ok=True)

    # Modify the rule to use specified delay
    rules = [
        GradualRampRule(feature='1', ramp_height=10.0, delay=delay, sigma=30.0),
    ]
    num_rules = len(rules)
    for rule in rules:
        rule.max_strength = 1 / num_rules

    # Generate data (copying from create_synthetic_dataset)
    start_frames = np.random.randint(0, 10, size=num_cells)
    end_frames = np.random.randint(num_frames//2, num_frames, size=num_cells)

    features = ['0', '1', '2']

    all_features = {name: np.empty((num_frames, num_cells), dtype=np.float32) for name in features}
    all_deaths = np.empty(num_cells, dtype=np.float32)
    cifs = np.empty((num_frames, num_cells), dtype=np.float32)
    pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    hazards = np.empty((num_frames, num_cells), dtype=np.float32)

    for c in tqdm(range(num_cells), desc=f'Generating cells (delay={delay})'):
        cell = Cell(num_frames)
        start = start_frames[c]
        end = end_frames[c]

        # Apply rules
        for rule in rules:
            rule.apply(cell)

        cell.hazards = np.clip(cell.hazards, a_min=0.0, a_max=1.0)
        pmf = cell._compute_pmf()
        cif = np.cumsum(pmf)

        cifs[:, c] = cif
        pmfs[:, c] = pmf
        hazards[:, c] = cell.hazards

        # Sample death time from cif
        u = np.random.rand()
        if u > np.max(cif):
            death_frame = np.nan
        else:
            death_frame = np.argmax(cif >= u)
            if death_frame > end:
                death_frame = np.nan

        if death_frame < start or death_frame > end:
            death_frame = np.nan

        all_deaths[c] = death_frame
        cell.apply_observation_window(start, end)

        for name in features:
            all_features[name][:, c] = cell[name]

    with h5py.File(temp_path, 'w') as f:
        grp = f.create_group('Cells/Phase')
        for name in features:
            grp.create_dataset(name, data=all_features[name], dtype=np.float32)
        grp.create_dataset('CellDeath', data=all_deaths[np.newaxis, :], dtype=np.float32)
        grp.create_dataset('CIFs', data=cifs, dtype=np.float32)
        grp.create_dataset('PMFs', data=pmfs, dtype=np.float32)
        grp.create_dataset('Hazards', data=hazards, dtype=np.float32)

    return temp_path


def analyze_delay_effect(delays=[100, 150, 200], num_cells=1000, num_frames=1000,
                         num_bins=5, min_length=200, max_time_to_death=100):
    """Analyze how different delay values affect the dataset using actual CellDataset class."""

    fig, axes = plt.subplots(len(delays), 4, figsize=(20, 5*len(delays)))
    if len(delays) == 1:
        axes = axes.reshape(1, -1)

    for delay_idx, delay in enumerate(delays):
        print(f"\n{'='*60}")
        print(f"Analyzing delay = {delay}")
        print(f"{'='*60}")

        # Create dataset with this delay
        dataset_path = create_dataset_with_delay(delay, num_cells, num_frames)

        # Create CellDataset exactly as in training
        features = ['0', '1', '2']
        dataset = CellDataset(
            hdf5_paths=[dataset_path],
            features=features,
            num_bins=num_bins,
            min_length=min_length,
            max_time_to_death=max_time_to_death,
            uncensored_only=False,
        )

        # Get normalization stats (as in training)
        means, stds = dataset.get_normalization_stats()

        print(f"\nDataset created:")
        print(f"  Total samples in metadata: {len(dataset)}")
        print(f"  Bin edges: {dataset.event_time_bins}")

        # Sample from dataset to get distribution
        num_samples = 10000
        time_to_events = []
        time_to_event_bins = []
        event_indicators = []
        valid_samples = 0
        none_samples = 0

        for _ in tqdm(range(num_samples), desc=f'Sampling dataset (delay={delay})'):
            idx = np.random.randint(len(dataset))
            item = dataset[idx]
            if item is None:
                none_samples += 1
                continue

            valid_samples += 1
            time_to_events.append(item['time_to_event'])
            time_to_event_bins.append(item['time_to_event_bin'])
            event_indicators.append(item['event_indicator'])

        time_to_events = np.array(time_to_events)
        time_to_event_bins = np.array(time_to_event_bins)
        event_indicators = np.array(event_indicators)

        # Filter for events only
        event_mask = event_indicators == 1
        event_time_to_events = time_to_events[event_mask]
        event_bins = time_to_event_bins[event_mask]

        print(f"\nSampling results:")
        print(f"  Valid samples: {valid_samples}/{num_samples}")
        print(f"  None samples: {none_samples}/{num_samples}")
        print(f"  Events: {np.sum(event_indicators)}")
        print(f"  Censored: {np.sum(1 - event_indicators)}")

        print(f"\nTime-to-event statistics (events only):")
        print(f"  Mean: {np.mean(event_time_to_events):.1f}")
        print(f"  Std: {np.std(event_time_to_events):.1f}")
        print(f"  Min: {np.min(event_time_to_events)}")
        print(f"  Max: {np.max(event_time_to_events)}")

        # Bin distribution
        bin_counts = np.bincount(event_bins, minlength=num_bins)
        print(f"\nSamples per bin:")
        for i in range(num_bins):
            print(f"  Bin {i}: {bin_counts[i]} ({100*bin_counts[i]/len(event_bins):.1f}%)")

        # Check for issues
        unique_bin_edges = len(np.unique(dataset.event_time_bins))
        if unique_bin_edges < num_bins + 1:
            print(f"\n⚠️  WARNING: Only {unique_bin_edges} unique bin edges (expected {num_bins+1})")

        max_bin_proportion = np.max(bin_counts) / len(event_bins) if len(event_bins) > 0 else 0
        if max_bin_proportion > 0.5:
            print(f"⚠️  WARNING: Most populous bin has {100*max_bin_proportion:.1f}% of samples")

        if np.min(bin_counts) == 0:
            empty_bins = np.where(bin_counts == 0)[0]
            print(f"⚠️  WARNING: Empty bins: {empty_bins}")

        # Plot 1: Hazard function (typical sample)
        with h5py.File(dataset_path, 'r') as f:
            typical_hazard = f['Cells/Phase/Hazards'][:, 0]
            typical_pmf = f['Cells/Phase/PMFs'][:, 0]

        ax = axes[delay_idx, 0]
        ax.plot(typical_hazard)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Saturation')
        ax.set_title(f'Delay={delay}: Hazard (sample 0)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Hazard')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: PMF
        ax = axes[delay_idx, 1]
        ax.plot(typical_pmf)
        ax.set_title(f'Delay={delay}: PMF (sample 0)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)

        # Plot 3: Time-to-event distribution
        ax = axes[delay_idx, 2]
        ax.hist(event_time_to_events, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(event_time_to_events), color='r', linestyle='--',
                   label=f'Mean={np.mean(event_time_to_events):.0f}')
        ax.set_title(f'Delay={delay}: Time-to-Event Distribution')
        ax.set_xlabel('Time-to-Event (frames)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Bin distribution
        ax = axes[delay_idx, 3]
        colors = ['red' if c == 0 else 'blue' for c in bin_counts]
        ax.bar(range(num_bins), bin_counts, alpha=0.7, edgecolor='black', color=colors)
        ax.set_title(f'Delay={delay}: Samples per Bin')
        ax.set_xlabel('Bin Index')
        ax.set_ylabel('Count')
        ax.set_xticks(range(num_bins))
        ax.grid(True, alpha=0.3, axis='y')

        # Add bin edges as text
        bin_text = f"Bin edges: {dataset.event_time_bins[:num_bins+1]}"
        ax.text(0.5, 0.95, bin_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = Path('temp') / 'delay_diagnostic_with_dataset.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n\nDiagnostic plot saved to {output_path}")
    plt.close()


def analyze_gaussian_scaling(num_frames=1000):
    """Analyze the Gaussian curve used for hazards."""
    from PhagoPred.survival_analysis.data.synthetic_data import get_gaussian_curve

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    sigmas = [10.0, 20.0, 30.0, 40.0]
    center = 500

    for sigma in sigmas:
        gaussian = get_gaussian_curve(num_frames, center, sigma)
        axes[0, 0].plot(gaussian, label=f'sigma={sigma}')

    axes[0, 0].set_title('Gaussian Curves (as used in code)')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Hazard Contribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Saturation')

    # Show the issue with scaling
    sigma = 30.0
    t = np.arange(num_frames)
    gaussian_raw = np.exp(-0.5 * ((t - center) / sigma)**2)
    gaussian_normalized = gaussian_raw / (sigma * np.sqrt(2 * np.pi))
    gaussian_scaled = gaussian_normalized * 5

    axes[0, 1].plot(gaussian_raw, label='Raw Gaussian')
    axes[0, 1].set_title(f'Raw Gaussian (sigma={sigma})')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(gaussian_normalized, label='PDF-normalized')
    axes[1, 0].set_title(f'PDF-normalized Gaussian (sigma={sigma})')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(gaussian_scaled, label='Scaled by 5 (current code)')
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Saturation threshold')
    axes[1, 1].set_title(f'Final Hazard Contribution (sigma={sigma})')
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Hazard')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    print(f"\nGaussian Analysis (sigma={sigma}, center={center}):")
    print(f"Max raw Gaussian: {np.max(gaussian_raw):.6f}")
    print(f"Max normalized Gaussian: {np.max(gaussian_normalized):.6f}")
    print(f"Max scaled Gaussian: {np.max(gaussian_scaled):.6f}")
    print(f"Integral of scaled Gaussian: {np.sum(gaussian_scaled):.6f}")

    plt.tight_layout()
    output_path = Path('temp') / 'gaussian_diagnostic.png'
    plt.savefig(output_path, dpi=150)
    print(f"Gaussian diagnostic plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    print("Running diagnostic analysis...")
    print("\n" + "="*60)
    print("PART 1: Analyzing Gaussian Scaling")
    print("="*60)
    analyze_gaussian_scaling()

    print("\n" + "="*60)
    print("PART 2: Analyzing Delay Effects with CellDataset")
    print("="*60)
    analyze_delay_effect(delays=[100, 150, 200], num_cells=1000, num_bins=5,
                        min_length=200, max_time_to_death=100)
