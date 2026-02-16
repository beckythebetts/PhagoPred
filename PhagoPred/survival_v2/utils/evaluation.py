from pathlib import Path
import os

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from PhagoPred.survival_v2.utils.metrics import concordance_index, compute_calibration_error, kl_divergence, integrated_brier_score
from PhagoPred.survival_v2.utils.plots import (
    visualise_prediction,
    plot_cm,
    plot_soft_confusion_matrix,
    plot_calibration_curve,
    plot_pmf_comparison_grid,
    plot_spread_analysis
)

def _compute_metrics_and_plots(
    all_pmf,
    all_true_bins,
    all_true_times,
    all_events,
    all_true_pmfs,
    bin_edges,
    save_dir=None
):
    """
    Shared function to compute evaluation metrics and generate plots.

    Args:
        all_pmf: numpy array of predicted PMFs (N, num_bins)
        all_true_bins: numpy array of true time bins (N,)
        all_true_times: numpy array of true event times (N,)
        all_events: numpy array of event indicators (N,)
        all_true_pmfs: numpy array of true PMFs (N, num_bins) or None
        bin_edges: array of bin edges
        save_dir: directory to save plots

    Returns:
        dict with evaluation metrics
    """
    num_bins = len(bin_edges) - 1

    # Filter to uncensored only for calculating accuracy
    uncensored_mask = all_events == 1
    pmf_unc = all_pmf[uncensored_mask]
    true_bins_unc = all_true_bins[uncensored_mask]

    # Predicted times (using all samples including censored)
    pred_bins_argmax = all_pmf.argmax(axis=1)
    time_bins_array = np.arange(num_bins)
    median_times = np.argmax((np.cumsum(all_pmf, axis=1) >= 0.5).astype(np.float64), axis=1)
    expected_times = (all_pmf * time_bins_array).sum(axis=1)

    # C-index (using all samples for proper concordance with censored data)
    c_index = concordance_index(
        all_pmf,
        all_true_times,
        all_events,
        bin_edges
    )

    # Accuracy (argmax on uncensored only)
    pred_bins_unc = pmf_unc.argmax(axis=1)
    accuracy = (pred_bins_unc == true_bins_unc).mean()

    # Brier score
    brier = integrated_brier_score(
        all_pmf,
        all_true_times,
        all_events,
        bin_edges
    )

    # Confusion matrix
    cm = confusion_matrix(true_bins_unc, pred_bins_unc, labels=np.arange(num_bins))
    if save_dir is not None:
        plot_cm(cm, save_path=save_dir / 'confusion_matrix.png')

    # Distribution-aware visualizations
    if save_dir is not None:
        # Soft confusion matrix (shows full PMF distributions)
        plot_soft_confusion_matrix(
            pmf_unc,
            true_bins_unc,
            save_path=save_dir / 'soft_confusion_matrix.png'
        )

        # Calibration curve
        plot_calibration_curve(
            pmf_unc,
            true_bins_unc,
            num_bins,
            save_path=save_dir / 'calibration_curve.png'
        )

        # Spread analysis (entropy/variance)
        true_pmf_unc = all_true_pmfs[uncensored_mask] if all_true_pmfs is not None and len(all_true_pmfs) > 0 else None
        plot_spread_analysis(
            pmf_unc,
            true_pmf_unc,
            true_bins_unc,
            save_path=save_dir / 'spread_analysis.png'
        )

        # PMF comparison grid (individual examples)
        plot_pmf_comparison_grid(
            pmf_unc,
            true_pmf_unc,
            true_bins_unc,
            bin_edges,
            num_examples=16,
            save_path=save_dir / 'pmf_comparison_grid.png'
        )

    # Calibration error
    calibration_error = compute_calibration_error(pmf_unc, true_bins_unc, num_bins)

    results = {
        'c_index': float(c_index),
        'accuracy': float(accuracy),
        'confusion_matrix': cm,
        'calibration_error': float(calibration_error),
        'brier_score': float(brier)
    }

    # Compute optimal metrics if true PMFs are available
    if all_true_pmfs is not None and len(all_true_pmfs) > 0:
        true_pmf_unc = all_true_pmfs[uncensored_mask]
        optimal_pred_bins = true_pmf_unc.argmax(axis=1)

        optimal_c_index = concordance_index(
            all_true_pmfs,
            all_true_times,
            all_events,
            bin_edges
        )

        optimal_brier = integrated_brier_score(
            all_true_pmfs,
            all_true_times,
            all_events,
            bin_edges
        )

        optimal_accuracy = (optimal_pred_bins == true_bins_unc).mean()

        results['optimal_c_index'] = float(optimal_c_index)
        results['optimal_brier_score'] = float(optimal_brier)
        results['optimal_accuracy'] = float(optimal_accuracy)

        kld = kl_divergence(pmf_unc, true_pmf_unc)
        results['kl_divergence'] = float(kld)

    return results


def evaluate_model(model, dataloader, device, visualise_predictions: int = 10, save_dir: Path = None):
    """
    Comprehensive evaluation of survival model.

    Args:
        model: SurvivalModel instance
        dataloader: data loader
        device: torch device
        num_bins: number of time bins
        visualise_predicitions: number of predictions to visualise

    Returns:
        dict with metrics:
            - c_index_median: C-index using median predicted time
            - c_index_expected: C-index using expected time
            - accuracy: prediction accuracy (argmax PMF vs true bin)
            - confusion_matrix: normalized confusion matrix
            - calibration_error: mean absolute calibration error
    """
    model.eval()
    model = model.to(device)

    all_pmf = []
    all_true_bins = []
    all_true_times = []
    all_events = []
    all_true_pmfs = []
    
    visualised_predictions_count = 0
    
    bin_edges = dataloader.dataset.event_time_bins
    num_bins = len(bin_edges) - 1
    feature_names = dataloader.dataset.features
    
    os.mkdir(save_dir) if save_dir is not None and not save_dir.exists() else None

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            model_output, attn_weights = model(batch['features'], batch['length'], mask=batch.get('mask'), return_attention=True)

            if isinstance(model_output, tuple):
                outputs = model_output[0]
            else:
                outputs = model_output

            pmf = model.predict_pmf(outputs)
            
            for i in range(batch['features'].size(0)):
                if visualised_predictions_count < visualise_predictions:
                    if batch['event_indicator'][i] == 0:
                        continue
                    if save_dir is not None:
                        save_path = save_dir / f'prediction_{visualised_predictions_count}.png'
                    else:
                        save_path = None
                        
                    single_sample = {batch_key: batch[batch_key][i] for batch_key in batch.keys()}
                    start_frame = single_sample['start_frame']
                    visualise_prediction(
                        single_sample,
                        predicted_pmf=pmf[i],
                        bin_edges=bin_edges,
                        # attn_weights=attn_weights[i],
                        feature_names=feature_names, 
                        save_path=save_path,
                        start_frame=start_frame,
                    )
                    visualised_predictions_count += 1
                    
            all_pmf.append(pmf.cpu())
            all_true_bins.append(batch['time_to_event_bin'].cpu())
            all_events.append(batch['event_indicator'].cpu())
            all_true_times.append(batch['time_to_event'].cpu())
            if 'binned_pmf' in batch:
                all_true_pmfs.append(batch['binned_pmf']) if 'binned_pmf' in batch else None
            

    # Concatenate
    all_pmf = torch.cat(all_pmf, dim=0).numpy()
    all_true_bins = torch.cat(all_true_bins, dim=0).numpy()
    all_events = torch.cat(all_events, dim=0).numpy()
    all_true_times = torch.cat(all_true_times, axis=0).numpy()
    all_true_pmfs = np.concatenate(all_true_pmfs, axis=0) if all_true_pmfs else None

    # Use shared function to compute metrics and generate plots
    results = _compute_metrics_and_plots(
        all_pmf=all_pmf,
        all_true_bins=all_true_bins,
        all_true_times=all_true_times,
        all_events=all_events,
        all_true_pmfs=all_true_pmfs,
        bin_edges=bin_edges,
        save_dir=save_dir
    )

    return results


def evaluate_classical_model(model, val_loader, device, visualise_predictions=10, save_dir=None):
    """
    Evaluate a classical ML model on the validation set.

    Args:
        model: ClassicalSurvivalModel instance
        val_loader: DataLoader for validation data
        device: Device (unused, for API compatibility)
        visualise_predictions: Number of predictions to visualize
        save_dir: Directory to save plots

    Returns:
        dict with evaluation metrics
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = val_loader.dataset.event_time_bins
    num_bins = len(bin_edges) - 1

    # Get predictions from classical model
    test_data = model.get_temporal_summary(val_loader.dataset)
    all_pmf = model.predict_pmfs(test_data)
    all_times = test_data['time_to_event']
    all_events = test_data['event_indicator']
    all_times_binned = test_data['time_to_event_bin']
    all_true_pmfs = test_data.get('binned_pmf', None)

    print(all_pmf.shape, all_times.shape, all_events.shape)

    # Generate individual prediction visualizations using shared function
    if visualise_predictions > 0 and save_dir is not None:
        feature_names = val_loader.dataset.features

        for i in range(min(visualise_predictions, len(all_pmf))):
            # Create sample dict compatible with visualise_prediction
            # Use raw_features from test_data to ensure deterministic features matching predictions
            sample = {
                'features': test_data['raw_features'][i],
                'time_to_event': all_times[i],
                'time_to_event_bin': all_times_binned[i],
                'event_indicator': all_events[i],
                'landmark_frame': test_data['landmark_frames'][i]
            }
            if all_true_pmfs is not None:
                sample['binned_pmf'] = all_true_pmfs[i]

            visualise_prediction(
                sample=sample,
                predicted_pmf=all_pmf[i],
                bin_edges=bin_edges,
                attn_weights=None,  # Classical models don't have attention
                feature_names=feature_names,
                save_path=save_dir / f'val_pred_{i+1}.png',
                start_frame=0,
            )

    # Use shared function to compute metrics and generate plots
    results = _compute_metrics_and_plots(
        all_pmf=all_pmf,
        all_true_bins=all_times_binned,
        all_true_times=all_times,
        all_events=all_events,
        all_true_pmfs=all_true_pmfs,
        bin_edges=bin_edges,
        save_dir=save_dir
    )

    # Add sample counts for classical models
    results['n_samples'] = len(all_times)
    results['n_events'] = int(all_events.sum())

    return results
