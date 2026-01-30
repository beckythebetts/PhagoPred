"""Analyze how often the feature ramp is visible in training samples using actual synthetic data."""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py

from PhagoPred.survival_analysis.data.synthetic_data import Cell, GradualRampRule
from PhagoPred.survival_analysis.data.dataset import CellDataset

# Parameters matching your training setup
num_cells = 1000
num_frames = 1000
min_length = 500
max_time_to_death = 100
num_bins = 5
ramp_length = 30

# Test different delay values
delays = [50, 100, 200, 500]

print("="*70)
print("RAMP VISIBILITY ANALYSIS (Using Real Synthetic Data)")
print("="*70)
print(f"Parameters:")
print(f"  num_cells: {num_cells}")
print(f"  num_frames: {num_frames}")
print(f"  min_length: {min_length}")
print(f"  max_time_to_death: {max_time_to_death}")
print(f"  num_bins: {num_bins}")
print(f"  ramp_length: {ramp_length}")
print(f"  Testing delays: {delays}")
print("="*70)

results = {}
temp_dir = Path('temp')
temp_dir.mkdir(exist_ok=True)

for delay in delays:
    print(f"\n{'='*70}")
    print(f"Creating synthetic dataset with delay = {delay} frames")
    print(f"{'='*70}")

    # Create synthetic dataset with this delay
    temp_path = temp_dir / f'synthetic_delay_{delay}.h5'

    # Generate synthetic data (similar to create_synthetic_dataset)
    rules = [
        GradualRampRule(feature='1', ramp_height=10.0, delay=delay, sigma=10.0),
    ]
    for rule in rules:
        rule.max_strength = 1.0 / len(rules)

    start_frames = np.random.randint(0, num_frames//2, size=num_cells)
    end_frames = np.random.randint(num_frames//2, num_frames, size=num_cells)

    features_list = ['0', '1', '2']
    all_features = {name: np.empty((num_frames, num_cells), dtype=np.float32) for name in features_list}
    all_deaths = np.empty(num_cells, dtype=np.float32)
    cifs = np.empty((num_frames, num_cells), dtype=np.float32)
    pmfs = np.empty((num_frames, num_cells), dtype=np.float32)
    hazards = np.empty((num_frames, num_cells), dtype=np.float32)
    ramp_starts = np.empty(num_cells, dtype=np.float32)
    ramp_ends = np.empty(num_cells, dtype=np.float32)

    for c in tqdm(range(num_cells), desc=f'Generating cells (delay={delay})'):
        cell = Cell(num_frames)
        start = start_frames[c]
        end = end_frames[c]

        # Track ramp timing
        ramp_start = np.random.randint(num_frames)
        ramp_starts[c] = ramp_start
        ramp_ends[c] = ramp_start + ramp_length

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

        for name in features_list:
            all_features[name][:, c] = cell[name]

    # Save to HDF5
    with h5py.File(temp_path, 'w') as f:
        grp = f.create_group('Cells/Phase')
        for name in features_list:
            grp.create_dataset(name, data=all_features[name], dtype=np.float32)
        grp.create_dataset('CellDeath', data=all_deaths[np.newaxis, :], dtype=np.float32)
        grp.create_dataset('CIFs', data=cifs, dtype=np.float32)
        grp.create_dataset('PMFs', data=pmfs, dtype=np.float32)
        grp.create_dataset('Hazards', data=hazards, dtype=np.float32)
        grp.create_dataset('RampStarts', data=ramp_starts[np.newaxis, :], dtype=np.float32)
        grp.create_dataset('RampEnds', data=ramp_ends[np.newaxis, :], dtype=np.float32)

    print(f"✓ Created synthetic dataset at {temp_path}")

    # Create CellDataset exactly as in training
    dataset = CellDataset(
        hdf5_paths=[temp_path],
        features=features_list,
        num_bins=num_bins,
        min_length=min_length,
        max_time_to_death=max_time_to_death,
        uncensored_only=False,
    )

    # Get normalization stats
    means, stds = dataset.get_normalization_stats()

    print(f"\n{'='*70}")
    print(f"Analyzing actual training samples (delay={delay})")
    print(f"{'='*70}")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Bin edges: {dataset.event_time_bins}")

    # Sample from dataset and analyze visibility
    num_samples = 5000
    valid_samples = 0
    none_samples = 0

    ramp_fully_visible = 0
    ramp_partially_visible = 0
    ramp_not_visible = 0

    # Track problematic cases
    events_with_ramp_visible = 0
    events_without_ramp_visible = 0
    censored_with_ramp_visible = 0
    censored_without_ramp_visible = 0

    time_to_events = []
    time_to_event_bins = []
    event_indicators = []
    landmark_to_ramp_start = []
    landmark_to_ramp_end = []

    for _ in tqdm(range(num_samples), desc=f'Sampling dataset (delay={delay})'):
        idx = np.random.randint(len(dataset))
        item = dataset[idx]

        if item is None:
            none_samples += 1
            continue

        valid_samples += 1

        # Get metadata
        cell_metadata = {key: dataset.cell_metadata[key][idx] for key in dataset.cell_metadata}
        file_idx = cell_metadata['File Idxs']
        local_cell_idx = cell_metadata['Local Cell Idxs']

        # Get ramp timing from HDF5
        with h5py.File(temp_path, 'r') as f:
            ramp_start = f['Cells/Phase/RampStarts'][0, local_cell_idx]
            ramp_end = f['Cells/Phase/RampEnds'][0, local_cell_idx]

        # Reconstruct landmark frame
        start_frame = cell_metadata['Start Frames']
        death_frame = cell_metadata['Death Frames']
        end_frame = cell_metadata['End Frames']

        last_frame = end_frame if np.isnan(death_frame) else death_frame
        event_indicator = 0 if np.isnan(death_frame) else 1

        if event_indicator == 0:
            min_landmark_dist = min_length
        else:
            min_landmark_dist = max(min_length, last_frame - max_time_to_death - start_frame)

        if last_frame <= start_frame + min_landmark_dist:
            continue

        # The actual landmark is randomly sampled, but we can estimate the range
        # For simplicity, use midpoint of possible landmark range
        landmark_frame = (start_frame + min_landmark_dist + last_frame) / 2

        # Check ramp visibility (ramp must be in [start_frame, landmark_frame])
        obs_start = start_frame
        obs_end = landmark_frame

        ramp_fully_in_obs = (ramp_start >= obs_start and ramp_end <= obs_end)
        ramp_partially_in_obs = not ramp_fully_in_obs and (
            (ramp_start >= obs_start and ramp_start <= obs_end) or
            (ramp_end >= obs_start and ramp_end <= obs_end) or
            (ramp_start < obs_start and ramp_end > obs_end)
        )
        ramp_not_in_obs = (ramp_start > obs_end)

        ramp_visible = ramp_fully_in_obs or ramp_partially_in_obs

        if ramp_fully_in_obs:
            ramp_fully_visible += 1
        elif ramp_partially_in_obs:
            ramp_partially_visible += 1
        else:
            ramp_not_visible += 1

        # Track problematic cases
        if event_indicator == 1:  # Death observed
            if ramp_visible:
                events_with_ramp_visible += 1
            else:
                events_without_ramp_visible += 1
        else:  # Censored
            if ramp_visible:
                censored_with_ramp_visible += 1
            else:
                censored_without_ramp_visible += 1

        # Collect statistics
        time_to_events.append(item['time_to_event'])
        time_to_event_bins.append(item['time_to_event_bin'])
        event_indicators.append(item['event_indicator'])
        landmark_to_ramp_start.append(landmark_frame - ramp_start)
        landmark_to_ramp_end.append(landmark_frame - ramp_end)

    # Calculate statistics
    time_to_events = np.array(time_to_events)
    time_to_event_bins = np.array(time_to_event_bins)
    event_indicators = np.array(event_indicators)

    event_mask = event_indicators == 1
    event_bins = time_to_event_bins[event_mask]

    pct_fully_visible = 100 * ramp_fully_visible / valid_samples if valid_samples > 0 else 0
    pct_partially_visible = 100 * ramp_partially_visible / valid_samples if valid_samples > 0 else 0
    pct_not_visible = 100 * ramp_not_visible / valid_samples if valid_samples > 0 else 0

    print(f"\nSampling Results:")
    print(f"  Valid samples: {valid_samples}/{num_samples}")
    print(f"  None samples: {none_samples}/{num_samples}")
    print(f"  Events: {np.sum(event_indicators)}")
    print(f"  Censored: {np.sum(1 - event_indicators)}")

    print(f"\nRamp Visibility:")
    print(f"  Ramp FULLY visible: {ramp_fully_visible:5d} ({pct_fully_visible:5.1f}%)")
    print(f"  Ramp PARTIALLY visible: {ramp_partially_visible:5d} ({pct_partially_visible:5.1f}%)")
    print(f"  Ramp NOT visible: {ramp_not_visible:5d} ({pct_not_visible:5.1f}%)")

    print(f"\n⚠ Problematic Cases (Death without visible ramp = noise):")
    total_events = events_with_ramp_visible + events_without_ramp_visible
    pct_events_with_ramp = 100 * events_with_ramp_visible / total_events if total_events > 0 else 0
    pct_events_without_ramp = 100 * events_without_ramp_visible / total_events if total_events > 0 else 0
    print(f"  Deaths WITH ramp visible: {events_with_ramp_visible:5d} ({pct_events_with_ramp:5.1f}%)")
    print(f"  Deaths WITHOUT ramp visible: {events_without_ramp_visible:5d} ({pct_events_without_ramp:5.1f}%) ⚠ NOISE")
    print(f"  Censored WITH ramp visible: {censored_with_ramp_visible:5d}")
    print(f"  Censored WITHOUT ramp visible: {censored_without_ramp_visible:5d}")

    print(f"\nTime-to-event statistics (events only):")
    if len(time_to_events[event_mask]) > 0:
        print(f"  Mean: {np.mean(time_to_events[event_mask]):.1f}")
        print(f"  Std: {np.std(time_to_events[event_mask]):.1f}")
        print(f"  Min: {np.min(time_to_events[event_mask])}")
        print(f"  Max: {np.max(time_to_events[event_mask])}")

    # Bin distribution
    if len(event_bins) > 0:
        bin_counts = np.bincount(event_bins, minlength=num_bins)
        print(f"\nSamples per bin (events only):")
        for i in range(num_bins):
            print(f"  Bin {i}: {bin_counts[i]:4d} ({100*bin_counts[i]/len(event_bins):5.1f}%)")

        # Check for issues
        max_bin_proportion = np.max(bin_counts) / len(event_bins)
        if max_bin_proportion > 0.5:
            print(f"\n⚠ WARNING: Most populous bin has {100*max_bin_proportion:.1f}% of samples")

        if np.min(bin_counts) == 0:
            empty_bins = np.where(bin_counts == 0)[0]
            print(f"⚠ WARNING: Empty bins: {empty_bins}")

    # Store results
    results[delay] = {
        'valid_samples': valid_samples,
        'ramp_fully_visible': ramp_fully_visible,
        'ramp_partially_visible': ramp_partially_visible,
        'ramp_not_visible': ramp_not_visible,
        'pct_fully_visible': pct_fully_visible,
        'pct_partially_visible': pct_partially_visible,
        'events_with_ramp_visible': events_with_ramp_visible,
        'events_without_ramp_visible': events_without_ramp_visible,
        'pct_events_without_ramp': pct_events_without_ramp,
        'landmark_to_ramp_start': landmark_to_ramp_start,
        'landmark_to_ramp_end': landmark_to_ramp_end,
        'bin_counts': bin_counts if len(event_bins) > 0 else np.zeros(num_bins),
        'time_to_events': time_to_events[event_mask] if len(event_bins) > 0 else [],
    }

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Ramp visibility percentage
ax = axes[0, 0]
x_pos = np.arange(len(delays))
fully_visible = [results[d]['pct_fully_visible'] for d in delays]
partially_visible = [results[d]['pct_partially_visible'] for d in delays]

ax.bar(x_pos - 0.2, fully_visible, width=0.4, label='Ramp Fully Visible', color='green', alpha=0.7)
ax.bar(x_pos + 0.2, partially_visible, width=0.4, label='Ramp Partially Visible', color='orange', alpha=0.7)
ax.set_xlabel('Delay (frames)', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage of Training Samples (%)', fontsize=13, fontweight='bold')
ax.set_title('Ramp Visibility in Actual Training Samples', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(delays)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')

# Add text annotations
for i, d in enumerate(delays):
    ax.text(i, fully_visible[i] + 2, f'{fully_visible[i]:.0f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# Plot 2: Deaths with/without ramp visible (THE CRITICAL METRIC)
ax = axes[0, 1]
deaths_with_ramp = [results[d]['events_with_ramp_visible'] for d in delays]
deaths_without_ramp = [results[d]['events_without_ramp_visible'] for d in delays]

x = np.arange(len(delays))
width = 0.35

bars1 = ax.bar(x - width/2, deaths_with_ramp, width, label='Deaths WITH ramp visible', color='green', alpha=0.7)
bars2 = ax.bar(x + width/2, deaths_without_ramp, width, label='Deaths WITHOUT ramp (NOISE)', color='red', alpha=0.7)

ax.set_xlabel('Delay (frames)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Event Samples', fontsize=13, fontweight='bold')
ax.set_title('⚠ Deaths With vs Without Visible Ramp\n(Red = Confusing Signal)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(delays)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add percentage annotations
for i, d in enumerate(delays):
    pct = results[d]['pct_events_without_ramp']
    total = deaths_with_ramp[i] + deaths_without_ramp[i]
    ax.text(i, total + 20, f'{pct:.0f}% noise', ha='center', fontweight='bold', fontsize=9, color='red')

# Plot 3: Bin distribution for each delay
ax = axes[0, 2]
bar_width = 0.8 / len(delays)
for i, delay in enumerate(delays):
    bin_counts = results[delay]['bin_counts']
    total = bin_counts.sum()
    if total > 0:
        bin_pcts = 100 * bin_counts / total
        x = np.arange(num_bins) + i * bar_width
        ax.bar(x, bin_pcts, width=bar_width, label=f'Delay={delay}', alpha=0.7)

ax.set_xlabel('Bin Index', fontsize=13, fontweight='bold')
ax.set_ylabel('Percentage of Events (%)', fontsize=13, fontweight='bold')
ax.set_title('Event Distribution Across Bins\n(Check for bin imbalance)', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(num_bins) + bar_width * len(delays) / 2)
ax.set_xticklabels(range(num_bins))
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Plot 4: Distribution of Landmark -> Ramp Start
ax = axes[1, 0]
colors = ['C0', 'C1', 'C2', 'C3', 'C4']
for i, delay in enumerate(delays):
    data = results[delay]['landmark_to_ramp_start']
    if len(data) > 0:
        ax.hist(data, bins=50, alpha=0.5, label=f'Delay={delay}', color=colors[i], edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Landmark time')
ax.set_xlabel('Landmark -> Ramp Start (frames)', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=13, fontweight='bold')
ax.set_title('Distribution: Landmark Position Relative to Ramp Start\n(Negative = Ramp in observation)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 5: Time-to-event distribution
ax = axes[1, 1]
for i, delay in enumerate(delays):
    data = results[delay]['time_to_events']
    if len(data) > 0:
        ax.hist(data, bins=30, alpha=0.5, label=f'Delay={delay}', color=colors[i], edgecolor='black')
ax.set_xlabel('Time-to-Event (frames)', fontsize=13, fontweight='bold')
ax.set_ylabel('Count', fontsize=13, fontweight='bold')
ax.set_title('Time-to-Event Distribution\n(From landmark to death)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axvline(x=max_time_to_death, color='red', linestyle='--', linewidth=2, label=f'max_time_to_death={max_time_to_death}')

# Plot 6: Noise percentage trend (Deaths without ramp)
ax = axes[1, 2]
noise_pcts = [results[d]['pct_events_without_ramp'] for d in delays]
ax.plot(delays, noise_pcts, marker='o', linewidth=3, markersize=10, color='red', label='Deaths w/o Ramp')
ax.fill_between(delays, 0, noise_pcts, alpha=0.3, color='red')
ax.set_xlabel('Delay (frames)', fontsize=13, fontweight='bold')
ax.set_ylabel('Noise Percentage (%)', fontsize=13, fontweight='bold')
ax.set_title('⚠ Training Noise: Deaths Without Visible Ramp\n(Lower is better)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='50% threshold', alpha=0.7)
ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='30% target', alpha=0.7)
ax.legend(fontsize=10)

# Add percentage labels
for i, d in enumerate(delays):
    ax.text(d, noise_pcts[i] + 2, f'{noise_pcts[i]:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10, color='red')

plt.tight_layout()
output_path = temp_dir / 'ramp_visibility_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'='*70}")
print(f"✓ Visualization saved to: {output_path}")
print(f"{'='*70}")

# Summary table
print(f"\n{'='*70}")
print("SUMMARY TABLE")
print(f"{'='*70}")
print(f"{'Delay':>6s} | {'Ramp Vis':>10s} | {'Death w/o Ramp':>15s} | {'Max Bin %':>11s} | {'Recommendation':>20s}")
print("-" * 90)
for delay in delays:
    fully_vis = results[delay]['pct_fully_visible']
    deaths_without_ramp = results[delay]['pct_events_without_ramp']
    bin_counts = results[delay]['bin_counts']
    max_bin_pct = 100 * np.max(bin_counts) / bin_counts.sum() if bin_counts.sum() > 0 else 0

    if fully_vis > 70 and deaths_without_ramp < 20 and max_bin_pct < 40:
        recommendation = "✓ EXCELLENT"
    elif fully_vis > 50 and deaths_without_ramp < 40 and max_bin_pct < 50:
        recommendation = "✓ GOOD"
    elif fully_vis > 30:
        recommendation = "⚠ MARGINAL"
    else:
        recommendation = "✗ POOR"

    print(f"{delay:6d} | {fully_vis:8.1f}% | {deaths_without_ramp:13.1f}% | {max_bin_pct:9.1f}% | {recommendation:>20s}")

print(f"{'='*90}")
print("\nKey Insights:")
print("  - 'Ramp Vis' = Percentage of samples with ramp fully visible")
print("  - 'Death w/o Ramp' = Deaths where ramp NOT visible (confusing signal)")
print("  - 'Max Bin %' = Percentage of events in most populous bin (check for imbalance)")
print("  - High 'Death w/o Ramp' means model sees deaths without the predictive feature")
print("  - This teaches the model to ignore features and predict a baseline distribution")
print(f"{'='*90}")
