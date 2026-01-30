from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams['font.family'] = 'serif'

# ==== PLOTTING RESULTS METRICS COMAPRISON ====
def plot_experiment_results(experiments_path: Path, plot_type: str = 'bar') -> None:
    """
    Plot accuray and c index for each experiment
    Args:
        experiment_path: Path to the experiment directory containing results.
        plot_type: Type of plot to create. Options: 'bar' (default) or 'box'
    """
    experiments = {
        'model': [],
        'attention': [],
        'loss': [],
        'dataset': [],
        'training': [],
        'feature_combo': [],
        'accuracy': [],
        'c_index': [],
        'optimal_accuracy': [],
        'optimal_c_index': [],
        'kl_divergence': [],
        'brier_score': [],
        'optimal_brier_score': [],
        }

    for experiment_path in experiments_path.iterdir():
        if experiment_path.is_dir():
            config = experiment_path / "config.json"
            results = experiment_path / "metrics.json"
            if config.exists() and results.exists():
                with open(config, 'r') as f:
                    config_data = json.load(f)
                with open(results, 'r') as f:
                    results_data = json.load(f)

                experiments['model'].append(config_data.get('model', 'N/A'))
                experiments['attention'].append(config_data.get('attention', 'N/A'))
                experiments['loss'].append(config_data.get('loss', 'N/A'))
                experiments['dataset'].append(config_data.get('dataset', 'N/A'))
                experiments['training'].append(config_data.get('training', 'N/A'))
                experiments['feature_combo'].append(config_data.get('feature_combo', 'N/A'))
                experiments['accuracy'].append(results_data.get('accuracy', 0))
                experiments['c_index'].append(results_data.get('c_index', 0))
                experiments['optimal_accuracy'].append(results_data.get('optimal_accuracy', 0))
                experiments['optimal_c_index'].append(results_data.get('optimal_c_index', 0))
                experiments['optimal_brier_score'].append(results_data.get('optimal_brier_score', 0))
                experiments['kl_divergence'].append(results_data.get('kl_divergence', 0))
                experiments['brier_score'].append(results_data.get('brier_score', 0))
                

    # Find which configurations vary
    unique_vals = {}
    varying_configs = []
    for config in ['model', 'attention', 'loss', 'dataset', 'training', 'feature_combo']:
        unique_vals[config] = np.unique(experiments[config])
        if len(unique_vals[config]) > 1:
            varying_configs.append(config)
    
    print(f"Varying configurations: {varying_configs}")

    if len(varying_configs) == 0:
        print("No varying configurations found")
        return
    elif len(varying_configs) == 1:
        # One independent variable
        if plot_type == 'box':
            fig = plot_boxplots(experiments, varying_configs[0], unique_vals[varying_configs[0]])
        else:
            fig = plot_grouped_bars(experiments, varying_configs[0], unique_vals[varying_configs[0]])
    elif len(varying_configs) == 2:
        # Two independent variables
        if plot_type == 'box':
            fig = plot_boxplots_two_vars(experiments, varying_configs[0], varying_configs[1],
                              unique_vals[varying_configs[0]], unique_vals[varying_configs[1]])
        else:
            fig = plot_grouped_bars_two_vars(experiments, varying_configs[0], varying_configs[1],
                              unique_vals[varying_configs[0]], unique_vals[varying_configs[1]])
    else:
        print(f"More than 2 varying configurations: {varying_configs}")
        print("Please filter experiments to have at most 2 varying parameters")
        return
        
    plt.savefig(experiments_path / "experiment_results.png", bbox_inches='tight')
    plt.close(fig)

def plot_grouped_bars(experiments: dict, var_name: str, unique_values: np.ndarray) -> plt.Figure:
    """Plot results with one independent variable using grouped bar chart."""
    # Calculate mean and std for each configuration
    config_stats = {}
    for val in unique_values:
        mask = np.array(experiments[var_name]) == val
        config_stats[val] = {
            'accuracy_mean': np.mean(np.array(experiments['accuracy'])[mask]),
            'accuracy_std': np.std(np.array(experiments['accuracy'])[mask]),
            'c_index_mean': np.mean(np.array(experiments['c_index'])[mask]),
            'c_index_std': np.std(np.array(experiments['c_index'])[mask]),
        }

    # Prepare data for plotting
    labels = [str(val) for val in unique_values]
    accuracy_means = [config_stats[val]['accuracy_mean'] for val in unique_values]
    accuracy_stds = [config_stats[val]['accuracy_std'] for val in unique_values]
    c_index_means = [config_stats[val]['c_index_mean'] for val in unique_values]
    c_index_stds = [config_stats[val]['c_index_std'] for val in unique_values]

    # Create grouped bar chart
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, accuracy_means, width, label='Accuracy',
                   yerr=accuracy_stds, capsize=5, alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, c_index_means, width, label='C-Index',
                   yerr=c_index_stds, capsize=5, alpha=0.8, color='#A23B72')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(var_name.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance Comparison by {var_name.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    return fig

def plot_grouped_bars_two_vars(experiments: dict, var1_name: str, var2_name: str,
                               var1_values: np.ndarray, var2_values: np.ndarray) -> plt.Figure:
    """Plot results with two independent variables using grouped bar chart with patterns."""
    # Calculate mean and std for each combination
    config_stats = {}
    for val1 in var1_values:
        for val2 in var2_values:
            mask = (np.array(experiments[var1_name]) == val1) & (np.array(experiments[var2_name]) == val2)
            if np.any(mask):
                key = (val1, val2)
                config_stats[key] = {
                    'accuracy_mean': np.mean(np.array(experiments['accuracy'])[mask]),
                    'accuracy_std': np.std(np.array(experiments['accuracy'])[mask]),
                    'c_index_mean': np.mean(np.array(experiments['c_index'])[mask]),
                    'c_index_std': np.std(np.array(experiments['c_index'])[mask]),
                }
    optimal_accuracy = np.mean(np.array(experiments['optimal_accuracy']))
    optimal_c_index = np.mean(np.array(experiments['optimal_c_index']))
    # Create subplots for accuracy and c-index
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colours = plt.get_cmap('Set1')

    # Set up x-axis positions
    n_var1 = len(var1_values)
    n_var2 = len(var2_values)
    width = (0.8 / n_var2)
    x = np.arange(n_var1)

    # Plot accuracy
    for j, val2 in enumerate(var2_values):
        accuracy_means = []
        accuracy_stds = []
        for val1 in var1_values:
            key = (val1, val2)
            if key in config_stats:
                accuracy_means.append(config_stats[key]['accuracy_mean'])
                accuracy_stds.append(config_stats[key]['accuracy_std'])
            else:
                accuracy_means.append(0)
                accuracy_stds.append(0)

        offset = ((j - n_var2/2 + 0.5) * width) + j*width*0.1
        bars = ax1.bar(x + offset, accuracy_means, width,
                        label=f'{var2_name}={val2}',
                        yerr=accuracy_stds, capsize=0, alpha=1.0,
                    #   color=[colours[i] for i in range(n_var1)],
                        color=colours(j),
                        # hatch=patterns[j], 
                        edgecolor='black', linewidth=1.0)

        ax1.axhline(optimal_accuracy, color='gray', linestyle='--', label='Optimal Valu' if j == 0 else "")

        # Add value labels on bars
        for bar, mean in zip(bars, accuracy_means):
            if mean > 0:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
                
    # Plot c-index
    for j, val2 in enumerate(var2_values):
        c_index_means = []
        c_index_stds = []
        for val1 in var1_values:
            key = (val1, val2)
            if key in config_stats:
                c_index_means.append(config_stats[key]['c_index_mean'])
                c_index_stds.append(config_stats[key]['c_index_std'])
            else:
                c_index_means.append(0)
                c_index_stds.append(0)

        offset = ((j - n_var2/2 + 0.5) * width) + j*width*0.1
        bars = ax2.bar(x + offset, c_index_means, width,
                      label=val2.replace("_", " ").capitalize(),
                      yerr=c_index_stds, capsize=0, alpha=1.0,
                    #   color=[colors[i] for i in range(n_var1)],
                      color = colours(j),
                    #   hatch=patterns[j],
                      edgecolor='black', 
                      linewidth=1.0)

       
        ax2.axhline(optimal_c_index, color='gray', linestyle='--', label='Optimal C-Index' if j == 0 else "")

        # Add value labels on bars
        for bar, mean in zip(bars, c_index_means):
            if mean > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
                
    # Configure accuracy plot
    ax1.set_xlabel(var1_name.capitalize(), fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v).replace('_', ' ').upper() for v in var1_values], rotation=0, ha='center')
    # ax1.grid(True, alpha=0.3, axis='y')
    # ax1.set_ylim(0, 1.1)

    # Configure c-index plot
    ax2.set_xlabel(var1_name.capitalize(), fontsize=12)
    ax2.set_ylabel('C-Index', fontsize=12)
    ax2.set_title('C-Index Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v).replace('_', ' ').upper() for v in var1_values], rotation=0, ha='center')
    # ax2.grid(True, alpha=0.3, axis='y')
    # ax2.set_ylim(0, 1.1)

    # Add single legend at the bottom with multiple columns
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, title=var2_name.capitalize(),
               fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(labels), 8), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)  # Make room for the legend at the bottom
    return fig

def plot_boxplots(experiments: dict, var_name: str, unique_values: np.ndarray) -> plt.Figure:
    """Plot results with one independent variable using box plots."""
    # Collect all values for each configuration
    config_data = {}
    metrics = ['accuracy', 'c_index','kl_divergence', 'brier_score']
    for val in unique_values:
        mask = np.array(experiments[var_name]) == val
        config_data[val] = {metric: np.array(experiments[metric])[mask].tolist() for metric in metrics}

    # Prepare data for plotting
    labels = [str(val) for val in unique_values]

    # Create grouped box plots
    x = np.arange(len(labels))
    width = 0.35
    
    cmap = plt.get_cmap('Set1')
    # colours = [cmap(i) for i in range(len(labels))]
    

    fig, axs = plt.subplots(1, len(metrics), figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        for j, var in enumerate(unique_values):
            data = config_data[var][metric]
            axs[i].boxplot(data, positions=[j], widths=width*0.6,
                                patch_artist=True, showmeans=True,
                                boxprops=dict(facecolor=cmap(j), alpha=0.7, edgecolor='black'),
                                medianprops=dict(color='black', linewidth=1.5),
                                meanprops=dict(marker='o', markerfacecolor='black',
                                                markeredgecolor='black', markersize=4),
                                whiskerprops=dict(color='black'),
                                capprops=dict(color='black'))
            axs[i].set_xlabel(var_name.capitalize(), fontsize=12)
            axs[i].set_ylabel(metric.capitalize(), fontsize=12)
            axs[i].set_xticks(x, labels,  ha='right')
            axs[i].set_title(f'{metric.capitalize()}', fontsize=14)
    
    handles = [Patch(facecolor=cmap(j), alpha=0.7, edgecolor='black', label=str(val)) 
               for j, val in enumerate(unique_values)]
    fig.legend(handles=handles, title=var_name.capitalize(),
               fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(labels), 8), frameon=True)

    plt.tight_layout()
    return fig

def plot_boxplots_two_vars(experiments: dict, var1_name: str, var2_name: str,
                           var1_values: np.ndarray, var2_values: np.ndarray) -> plt.Figure:
    """Plot results with two independent variables using box plots."""
    # Collect all values for each combination
    config_data = {}
    metrics = ['accuracy', 'c_index', 'kl_divergence', 'brier_score']
    for val1 in var1_values:
        for val2 in var2_values:
            mask = (np.array(experiments[var1_name]) == val1) & (np.array(experiments[var2_name]) == val2)
            if np.any(mask):
                key = (val1, val2)
                # config_data[key] = {
                #     'accuracy': np.array(experiments['accuracy'])[mask].tolist(),
                #     'c_index': np.array(experiments['c_index'])[mask].tolist(),
                # }
                config_data[key] = {metric: np.array(experiments[metric])[mask].tolist() for metric in metrics}

    # optimal_accuracy = np.mean(np.array(experiments['optimal_accuracy']))
    # optimal_c_index = np.mean(np.array(experiments['optimal_c_index']))

    # Create subplots for accuracy and c-index
    fig, axs = plt.subplots(1, len(metrics), figsize=(16, 6))

    colours = plt.get_cmap('Set1')

    # Set up x-axis positions
    n_var1 = len(var1_values)
    n_var2 = len(var2_values)
    width = (0.8 / n_var2)
    x = np.arange(n_var1)

    for i, metric in enumerate(metrics):
        for j, val2 in enumerate(var2_values):
            data = []
            positions = []
            for k, val1 in enumerate(var1_values):
                key = (val1, val2)
                if key in config_data and len(config_data[key][metric]) > 0:
                    data.append(config_data[key][metric])
                    offset = ((j - n_var2/2 + 0.5) * width) + j*width*0.1
                    positions.append(x[k] + offset)

            if data:
                bp = axs[i].boxplot(data, positions=positions, widths=width*0.6,
                               patch_artist=True, showmeans=True,
                               boxprops=dict(facecolor=colours(j), alpha=0.7, edgecolor='black'),
                               medianprops=dict(color='black', linewidth=1.5),
                               meanprops=dict(marker='o', markerfacecolor='black',
                                            markeredgecolor='black', markersize=4),
                               whiskerprops=dict(color='black'),
                               capprops=dict(color='black'))

                # Create legend entry
                axs[i].plot([], [], color=colours(j), linewidth=10, alpha=0.7,
                        label=val2.replace("_", " ").capitalize())
                axs[i].set_xticks(x)
                axs[i].set_xticklabels([str(v).replace('_', ' ').capitalize() for v in var1_values], rotation=0, ha='center')
                axs[i].set_xlabel(var1_name.capitalize(), fontsize=12)
                axs[i].set_ylabel(metric.capitalize().replace('_', ' '), fontsize=12)
                axs[i].set_title(f"{metric.capitalize().replace('_', ' ')} Comparison", fontsize=14)
                axs[i].grid(True, alpha=0.3, axis='y')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=var2_name.capitalize(),
               fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(len(labels), 8), frameon=True)

    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.15)  # Make room for the legend at the bottom
    return fig

# ==== PLOTTIGS CONFUSION MATRICES ====

def plot_confusion_matrices(experiments_path: Path) -> None:
    """
    Plot confusion matrices for experiments, averaged over repeats with standard deviations.

    Args:
        experiments_path: Path to the experiment directory containing results.
    """
    experiments = {
        'model': [],
        'attention': [],
        'loss': [],
        'dataset': [],
        'training': [],
        'feature_combo': [],
        'confusion_matrix': [],
    }

    for experiment_path in experiments_path.iterdir():
        if experiment_path.is_dir():
            config = experiment_path / "config.json"
            results = experiment_path / "metrics.json"
            if config.exists() and results.exists():
                with open(config, 'r') as f:
                    config_data = json.load(f)
                with open(results, 'r') as f:
                    results_data = json.load(f)

                experiments['model'].append(config_data.get('model', 'N/A'))
                experiments['attention'].append(config_data.get('attention', 'N/A'))
                experiments['loss'].append(config_data.get('loss', 'N/A'))
                experiments['dataset'].append(config_data.get('dataset', 'N/A'))
                experiments['training'].append(config_data.get('training', 'N/A'))
                experiments['feature_combo'].append(config_data.get('feature_combo', 'N/A'))

                cm = results_data.get('confusion_matrix', None)
                if cm is not None:
                    experiments['confusion_matrix'].append(np.array(cm))

    # Find which configurations vary
    unique_vals = {}
    varying_configs = []
    for config in ['model', 'attention', 'loss', 'dataset', 'training', 'feature_combo']:
        unique_vals[config] = np.unique(experiments[config])
        if len(unique_vals[config]) > 1:
            varying_configs.append(config)

    print(f"Varying configurations for confusion matrices: {varying_configs}")

    if len(varying_configs) == 0:
        print("No varying configurations found")
        return
    elif len(varying_configs) == 1:
        # One independent variable - plot in a line
        fig = plot_cms_one_var(experiments, varying_configs[0], unique_vals[varying_configs[0]])
    elif len(varying_configs) == 2:
        # Two independent variables - plot in a grid
        fig = plot_cms_two_vars(experiments, varying_configs[0], varying_configs[1],
                                 unique_vals[varying_configs[0]], unique_vals[varying_configs[1]])
    else:
        print(f"More than 2 varying configurations: {varying_configs}")
        print("Please filter experiments to have at most 2 varying parameters")
        return

    plt.savefig(experiments_path / "confusion_matrices.png", bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_cms_one_var(experiments: dict, var_name: str, unique_values: np.ndarray) -> plt.Figure:
    """Plot confusion matrices for one independent variable in a horizontal line."""
    n_configs = len(unique_values)

    # Calculate average confusion matrices and stds
    avg_cms = {}
    std_cms = {}

    for val in unique_values:
        mask = np.array(experiments[var_name]) == val
        cms_for_config = [cm for cm, m in zip(experiments['confusion_matrix'], mask) if m]

        if cms_for_config:
            avg_cms[val] = np.mean(cms_for_config, axis=0)
            std_cms[val] = np.std(cms_for_config, axis=0)
        else:
            # Handle case with no data
            avg_cms[val] = None
            std_cms[val] = None

    # Normalize confusion matrices
    normalized_cms = {}
    for val in unique_values:
        if avg_cms[val] is not None:
            row_sums = avg_cms[val].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            normalized_cms[val] = avg_cms[val] / row_sums
        else:
            normalized_cms[val] = None

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 5))
    if n_configs == 1:
        axes = [axes]

    for idx, val in enumerate(unique_values):
        ax = axes[idx]

        if normalized_cms[val] is not None:
            cm = normalized_cms[val]

            # Plot heatmap
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Proportion', rotation=270, labelpad=15, fontsize=10)

            # Add text annotations with mean ± std
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    mean_val = cm[i, j]
                    std_val = std_cms[val][i, j] / (avg_cms[val][i, :].sum() + 1e-10)  # Normalized std

                    text_color = 'white' if mean_val > 0.5 else 'black'
                    ax.text(j, i, f'{mean_val:.2f}\n±{std_val:.2f}',
                           ha='center', va='center', color=text_color, fontsize=8)

            ax.set_title(f'{var_name}={str(val).replace("_", " ").capitalize()}',
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{var_name}={val}', fontsize=12, fontweight='bold')

        ax.set_xlabel('Predicted Bin', fontsize=10)
        ax.set_ylabel('True Bin', fontsize=10)
        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))

    plt.suptitle(f'Confusion Matrices by {var_name.capitalize()}',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_cms_two_vars(experiments: dict, var1_name: str, var2_name: str,
                      var1_values: np.ndarray, var2_values: np.ndarray) -> plt.Figure:
    """Plot confusion matrices for two independent variables in a grid."""
    n_var1 = len(var1_values)
    n_var2 = len(var2_values)

    # Calculate average confusion matrices and stds for each combination
    avg_cms = {}
    std_cms = {}

    for val1 in var1_values:
        for val2 in var2_values:
            mask = (np.array(experiments[var1_name]) == val1) & (np.array(experiments[var2_name]) == val2)
            cms_for_config = [cm for cm, m in zip(experiments['confusion_matrix'], mask) if m]

            key = (val1, val2)
            if cms_for_config:
                avg_cms[key] = np.mean(cms_for_config, axis=0)
                std_cms[key] = np.std(cms_for_config, axis=0)
            else:
                avg_cms[key] = None
                std_cms[key] = None

    # Normalize confusion matrices
    normalized_cms = {}
    for key in avg_cms:
        if avg_cms[key] is not None:
            row_sums = avg_cms[key].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            normalized_cms[key] = avg_cms[key] / row_sums
        else:
            normalized_cms[key] = None

    # Create grid of subplots
    fig, axes = plt.subplots(n_var2, n_var1, figsize=(5 * n_var1, 5 * n_var2))
    if n_var1 == 1 and n_var2 == 1:
        axes = np.array([[axes]])
    elif n_var1 == 1:
        axes = axes.reshape(-1, 1)
    elif n_var2 == 1:
        axes = axes.reshape(1, -1)

    for i, val2 in enumerate(var2_values):
        for j, val1 in enumerate(var1_values):
            ax = axes[i, j]
            key = (val1, val2)

            if normalized_cms[key] is not None:
                cm = normalized_cms[key]

                # Plot heatmap
                im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Proportion', rotation=270, labelpad=15, fontsize=9)

                # Add text annotations with mean ± std
                for row in range(cm.shape[0]):
                    for col in range(cm.shape[1]):
                        mean_val = cm[row, col]
                        std_val = std_cms[key][row, col] / (avg_cms[key][row, :].sum() + 1e-10)  # Normalized std

                        text_color = 'white' if mean_val > 0.5 else 'black'
                        ax.text(col, row, f'{mean_val:.2f}\n±{std_val:.2f}',
                               ha='center', va='center', color=text_color, fontsize=7)

                title = f'{var1_name}={str(val1).replace("_", " ").capitalize()}\n{var2_name}={str(val2).replace("_", " ").capitalize()}'
                ax.set_title(title, fontsize=10, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{var1_name}={val1}\n{var2_name}={val2}', fontsize=10)

            # Labels
            if i == n_var2 - 1:
                ax.set_xlabel('Predicted Bin', fontsize=9)
            if j == 0:
                ax.set_ylabel('True Bin', fontsize=9)

            if normalized_cms[key] is not None:
                ax.set_xticks(np.arange(cm.shape[1]))
                ax.set_yticks(np.arange(cm.shape[0]))

    plt.suptitle(f'Confusion Matrices: {var1_name.capitalize()} vs {var2_name.capitalize()}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    return fig

# ==== PLOTTING LOSSES ====
def plot_experiment_losses(experiments_path: Path, metric: str = 'total') -> None:
    """
    Plot training and validation losses over epochs for each experiment.
    Each unique configuration gets its own color, with confidence intervals from repeats.

    Args:
        experiments_path: Path to the experiment directory containing results.
        metric: Which metric to plot from training history. Options: 'total', 'nll', 'ranking', 'prediction'
    """
    experiments = {
        'model': [],
        'attention': [],
        'loss': [],
        'dataset': [],
        'training': [],
        'feature_combo': [],
        'train_losses': [],
        'val_losses': [],
    }

    # Load all experiment data
    for experiment_path in experiments_path.iterdir():
        if experiment_path.is_dir():
            config = experiment_path / "config.json"
            history = experiment_path / "training_history.json"
            if config.exists() and history.exists():
                with open(config, 'r') as f:
                    config_data = json.load(f)
                with open(history, 'r') as f:
                    history_data = json.load(f)

                experiments['model'].append(config_data.get('model', 'N/A'))
                experiments['attention'].append(config_data.get('attention', 'N/A'))
                experiments['loss'].append(config_data.get('loss', 'N/A'))
                experiments['dataset'].append(config_data.get('dataset', 'N/A'))
                experiments['training'].append(config_data.get('training', 'N/A'))
                experiments['feature_combo'].append(config_data.get('feature_combo', 'N/A'))

                # Extract loss curves
                train_losses = [epoch_data['train'][metric] for epoch_data in history_data]
                val_losses = [epoch_data['val'][metric] for epoch_data in history_data]
                experiments['train_losses'].append(train_losses)
                experiments['val_losses'].append(val_losses)

    # Find which configurations vary
    unique_vals = {}
    varying_configs = []
    for config in ['model', 'attention', 'loss', 'dataset', 'training', 'feature_combo']:
        unique_vals[config] = np.unique(experiments[config])
        if len(unique_vals[config]) > 1:
            varying_configs.append(config)

    print(f"Varying configurations for training curves: {varying_configs}")

    if len(varying_configs) == 0:
        print("No varying configurations found")
        return

    # Create unique configuration identifiers
    config_groups = {}
    for i in range(len(experiments['model'])):
        # Create a key from all varying configs
        key_parts = []
        for var_config in varying_configs:
            val = experiments[var_config][i]
            key_parts.append(f"{val}")
        config_key = "_".join(key_parts)

        if config_key not in config_groups:
            config_groups[config_key] = {'train': [], 'val': []}

        config_groups[config_key]['train'].append(experiments['train_losses'][i])
        config_groups[config_key]['val'].append(experiments['val_losses'][i])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colours = plt.get_cmap('tab10')

    for idx, (config_key, losses) in enumerate(sorted(config_groups.items())):
        color = colours(idx % 10)

        # Format label nicely
        label = config_key.replace('_', ' ').capitalize()

        # Training losses
        train_losses_array = np.array(losses['train'])  # (n_repeats, n_epochs)
        n_epochs = train_losses_array.shape[1]
        epochs = np.arange(1, n_epochs + 1)

        train_mean = np.mean(train_losses_array, axis=0)
        train_std = np.std(train_losses_array, axis=0)

        ax1.plot(epochs, train_mean, label=label, color=color, linewidth=2)
        ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                         alpha=0.3, color=color, edgecolor=None)

        # Validation losses
        val_losses_array = np.array(losses['val'])  # (n_repeats, n_epochs)
        val_mean = np.mean(val_losses_array, axis=0)
        val_std = np.std(val_losses_array, axis=0)

        ax2.plot(epochs, val_mean, label=label, color=color, linewidth=2)
        ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                         alpha=0.3, color=color, edgecolor=None)

    # Configure training plot
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Training {metric.capitalize()} Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')

    # Configure validation plot
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel(f'Validation {metric.capitalize()} Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig(experiments_path / f"training_curves_{metric}.png", bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {experiments_path / f'training_curves_{metric}.png'}")


if __name__ == "__main__":
    experiments_path = Path('/home/ubuntu/PhagoPred/PhagoPred/survival_v2/experiments/results/start_frame_feature_comparison_20260120_152554')

    # Plot main experiment results (accuracy, c-index, etc.)
    plot_experiment_results(experiments_path, plot_type='box')

    # Plot confusion matrices
    plot_confusion_matrices(experiments_path)
    
    plot_experiment_losses(experiments_path, metric='total')
