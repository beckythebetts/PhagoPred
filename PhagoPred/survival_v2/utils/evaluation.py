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
                    visualise_prediction(
                        single_sample,
                        predicted_pmf=pmf[i],
                        bin_edges=bin_edges,
                        attn_weights=attn_weights[i],
                        feature_names=feature_names, 
                        save_path=save_path
                    )
                    visualised_predictions_count += 1
                    
            all_pmf.append(pmf.cpu())
            all_true_bins.append(batch['time_to_event_bin'].cpu())
            all_events.append(batch['event_indicator'].cpu())
            all_true_times.append(batch['time_to_event'].cpu())
            if 'binned_pmf' in batch:
                all_true_pmfs.append(batch['binned_pmf']) if 'binned_pmf' in batch else None
            

    # Concatenate
    all_pmf = torch.cat(all_pmf, dim=0)
    all_true_bins = torch.cat(all_true_bins, dim=0)
    all_events = torch.cat(all_events, dim=0)
    all_true_times = torch.cat(all_true_times, axis=0)
    all_true_pmfs = np.concatenate(all_true_pmfs, axis=0) if all_true_pmfs is not None else None
    
    
    # Filter to uncensored only for calculating accurcay
    uncensored_mask = all_events == 1
    pmf_unc = all_pmf[uncensored_mask]
    true_bins_unc = all_true_bins[uncensored_mask].numpy()
    
    # Predicted times (using all samples including censored)
    pred_bins_argmax = all_pmf.argmax(dim=1).numpy()
    time_bins_tensor = torch.arange(num_bins).float()
    median_times = torch.argmax((torch.cumsum(all_pmf, dim=1) >= 0.5).float(), dim=1).numpy()
    expected_times = (all_pmf * time_bins_tensor).sum(dim=1).numpy()

    # C-index (using all samples for proper concordance with censored data)
    # c_index_argmax = concordance_index(pred_bins_argmax, all_true_bins.numpy(), all_events.numpy())
    # c_index_median = concordance_index(median_times, all_true_bins.numpy(), all_events.numpy())
    # c_index_expected = concordance_index(expected_times, all_true_bins.numpy(), all_events.numpy())
    c_index = concordance_index(
        all_pmf.numpy(), 
        all_true_times.numpy(), 
        all_events.numpy(), 
        bin_edges
        )

    # Accuracy (argmax on uncensored only)
    pred_bins_unc = pmf_unc.argmax(dim=1).numpy()
    accuracy = (pred_bins_unc == true_bins_unc).mean()


    # Brier score - using binned version for discrete time survival
    # brier = brier_score_binned(
    #     all_pmf.numpy(),
    #     all_true_bins.numpy(),
    #     all_events.numpy()
    # )
    brier = integrated_brier_score(
        all_pmf.numpy(),
        all_true_times.numpy(),
        all_events.numpy(),
        bin_edges
    )
    

    # Confusion matrix
    cm = confusion_matrix(true_bins_unc, pred_bins_unc, labels=np.arange(num_bins))
    plot_cm(cm, save_path=save_dir / 'confusion_matrix.png' if save_dir is not None else None)

    # Distribution-aware visualizations
    if save_dir is not None:
        # Soft confusion matrix (shows full PMF distributions)
        plot_soft_confusion_matrix(
            pmf_unc.numpy(),
            true_bins_unc,
            save_path=save_dir / 'soft_confusion_matrix.png'
        )

        # Calibration curve
        plot_calibration_curve(
            pmf_unc.numpy(),
            true_bins_unc,
            num_bins,
            save_path=save_dir / 'calibration_curve.png'
        )

        # Spread analysis (entropy/variance)
        true_pmf_unc = all_true_pmfs[uncensored_mask] if len(all_true_pmfs) > 0 else None
        plot_spread_analysis(
            pmf_unc.numpy(),
            true_pmf_unc,
            true_bins_unc,
            save_path=save_dir / 'spread_analysis.png'
        )

        # PMF comparison grid (individual examples)
        plot_pmf_comparison_grid(
            pmf_unc.numpy(),
            true_pmf_unc,
            true_bins_unc,
            bin_edges,
            num_examples=16,
            save_path=save_dir / 'pmf_comparison_grid.png'
        )

    # Calibration error
    calibration_error = compute_calibration_error(pmf_unc.numpy(), true_bins_unc, num_bins)

    results = {
        # 'c_index_argmax': c_index_argmax,
        # 'c_index_median': c_index_median,
        # 'c_index_expected': c_index_expected,
        'c_index': c_index,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'calibration_error': calibration_error,
        'brier_score': brier
    }

    if len(all_true_pmfs) > 0:
        true_pmf_unc = all_true_pmfs[uncensored_mask]
        optimal_pred_bins = true_pmf_unc.argmax(axis=1)
        # optimal_median_times = np.argmax((np.cumsum(true_pmf_unc, axis=1) >= 0.5).astype(np.float64), axis=1)
        # optimal_expected_times = (true_pmf_unc * time_bins_tensor.cpu().numpy()).sum(axis=1)

        # c_index_optimal_argmax = concordance_index(optimal_pred_bins, true_bins_unc, np.ones_like(true_bins_unc))
        # c_index_optimal_median = concordance_index(optimal_median_times, true_bins_unc, np.ones_like(true_bins_unc))
        # c_index_optimal_expected = concordance_index(optimal_expected_times, true_bins_unc, np.ones_like(true_bins_unc))

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

        results['optimal_c_index'] = optimal_c_index
        results['optimal_brier_score'] = optimal_brier
        results['optimal_accuracy'] = optimal_accuracy

        kld = kl_divergence(pmf_unc, true_pmf_unc)
        results['kl_divergence'] = kld

    return results
