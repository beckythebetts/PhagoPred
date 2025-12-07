import torch
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

from PhagoPred.survival_analysis.models.dynamic_deephit import model
from PhagoPred.survival_analysis.data.dataset import CellDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shap_deep_hit(model, X_background, X_test, num_bins=None, device='cpu'):
    """
    Compute SHAP values for a DeepHit model across multiple time bins.

    Args:
        model: PyTorch model returning [batch_size, num_bins] CIF probabilities
        X_background: [num_background, num_features] tensor for SHAP reference
        X_test: [num_test, num_features] tensor for SHAP explanation
        num_bins: number of discrete time bins (default: inferred from model output)
        device: 'cpu' or 'cuda'

    Returns:
        shap_values: list of [num_test, num_features] arrays, one per time bin
    """
    model.eval()
    model.to(device)
    X_background = X_background.to(device)
    X_test = X_test.to(device)

    # Infer number of bins
    if num_bins is None:
        with torch.no_grad():
            num_bins = model(X_test[:1]).shape[1]

    shap_values_all = []

    for t_idx in range(num_bins):
        # Model output: CIF at time t_idx
        def model_cif_at_t(X_input):
            return model(X_input)[:, t_idx]

        # Use DeepExplainer
        explainer = shap.DeepExplainer(model_cif_at_t, X_background)
        shap_vals = explainer.shap_values(X_test)
        shap_values_all.append(shap_vals)  # shape: [num_test, num_features]

    return shap_values_all

def plot_shap_heatmap(shap_values_all, feature_names=None):
    """
    Plot SHAP values as a heatmap (time bins x features)
    shap_values_all: list of [num_test, num_features], one per time bin
    """
    # Average across samples
    shap_array = np.mean(np.stack(shap_values_all, axis=0), axis=1)  # [num_bins, num_features]

    plt.figure(figsize=(12,6))
    plt.imshow(shap_array.T, aspect='auto', cmap='bwr', interpolation='nearest')
    plt.colorbar(label='Mean SHAP value')
    plt.xlabel('Time bin')
    plt.ylabel('Feature')
    if feature_names is not None:
        plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.title('SHAP values across time bins')
    plt.show()

if __name__ == '__main__':
    model_dir = Path('/home/ubuntu/PhagoPred/PhagoPred/survival_analysis/models/dynamic_deephit/test_run')
    hdf5_paths = [Path('PhagoPred')/'Datasets'/'synthetic.h5']
    
    with open(model_dir / 'model_params.json', 'r') as f:
        model_params = json.load(f)

    model_ = model.DynamicDeepHit
    features = model_params['features']
    model_ = model_(**model_params)
    model_.load_state_dict(torch.load(model_dir / 'model.pth', map_location=device))
    
    normalisation_means = model_params['normalization_means']
    normalization_stds = model_params['normalization_stds']
    dataset = CellDataset(
        hdf5_paths=hdf5_paths,
        features=features,
        means=np.array(normalisation_means),
        stds=np.array(normalization_stds),
        num_bins=model_params['output_size'],
        event_time_bins=np.array(model_params['event_time_bins']),
        uncensored_only=False,
        min_length=100,
        max_time_to_death=200,
    )
    
    
    shap_values_all = shap_deep_hit(model_, )
    feature_names = ['age','sex','blood_pressure','cholesterol']  # example
    plot_shap_heatmap(shap_values_all, feature_names)