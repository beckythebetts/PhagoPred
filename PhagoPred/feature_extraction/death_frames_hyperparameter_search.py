import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import h5py

from PhagoPred.feature_extraction.extract_features import extract_features
from PhagoPred.feature_extraction import features
from PhagoPred import SETTINGS


def cell_death_hyperparameter_search(dataset: Path, true_deaths_txt: Path, save_csv: bool = False, csv_path: Path = None) -> pd.DataFrame:
    # Load true deaths (NaNs preserved)
    true_deaths = np.genfromtxt(true_deaths_txt, delimiter=',', usecols=1)

    thresholds = np.round(np.arange(0.5, 1.05, 0.05), 2)
    smoothed_frames = np.round(np.arange(1, 25, 1), 0)
    # min_frames_deads = np.arange(5, 30, 5)

    results = []

    # Optional: show progress bar
    total_iters = len(thresholds) * len(smoothed_frames)
    pbar = tqdm(total=total_iters, desc="Hyperparam search")

    for threshold in thresholds:
        # for min_frames_dead in min_frames_deads:
        for smoothed_frame in smoothed_frames:
            # Call your extraction function that updates the dataset with the predicted deaths
            extract_features(dataset, phase_features=[features.CellDeath(threshold, 0, smoothed_frame)])

            with h5py.File(dataset, 'r') as f:
                estimated_deaths = f['Cells']['Phase']['CellDeath'][0, :len(true_deaths)]

            has_death_mask = ~np.isnan(true_deaths)
            no_death_mask = np.isnan(true_deaths)

            # Compute MAE on cells with true death frames and predicted death frames
            valid_mask = has_death_mask & ~np.isnan(estimated_deaths)
            mae = np.nan
            top_error_idxs = []
            top_frames = []
            if np.any(valid_mask):
                errors = np.abs(estimated_deaths[valid_mask] - true_deaths[valid_mask])
                mae = np.mean(errors)
                
                top3_idxs = np.argsort(errors)[-3:]
                top3_idxs = np.where(valid_mask)[0][top3_idxs].tolist()
                top3_pred_frames = estimated_deaths[top3_idxs]

            # Compute accuracy for no-death cells (fraction of predicted NaNs)
            no_death_accuracy = np.nan
        

            # Create binary labels
            true_labels = ~np.isnan(true_deaths)           # True = death, False = no-death
            pred_labels = ~np.isnan(estimated_deaths)      # True = predicted death

            # Check if there are any labels at all to evaluate
            if np.any(true_labels) or np.any(~true_labels):
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                recall = recall_score(true_labels, pred_labels, zero_division=0)
                f1 = f1_score(true_labels, pred_labels, zero_division=0)
            else:
                precision = recall = f1 = np.nan  # No data to compute on

            results.append({
                'threshold': threshold,
                # 'min_frames_dead': min_frames_dead,
                'smoothed_frame': smoothed_frame,
                'top3_idxs': top3_idxs,
                'top3_pred_frame': top3_pred_frames,
                'mae_death_cells': mae,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            print(mae, f1)

            pbar.update(1)

    pbar.close()

    results_df = pd.DataFrame(results)

    if save_csv:
        if csv_path is None:
            csv_path = Path("cell_death_hyperparam_search_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return results_df


def plot_hyperparam_heatmaps(results_path_or_df, save_dir='hyperparam_heatmaps'):
    # Load results
    if isinstance(results_path_or_df, pd.DataFrame):
        df = results_path_or_df
    else:
        df = pd.read_csv(results_path_or_df)

    print(df.dtypes)
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Print best rows for each metric
    print("Best row by MAE (lowest):")
    print(df.loc[df['mae_death_cells'].idxmin()])
    print("\nBest row by No-death F1-Score (highest):")
    print(df.loc[df['f1_score'].idxmax()])

    # List of hyperparameters (exclude performance columns)
    # param_cols = [c for c in df.columns if c not in ['mae_death_cells', 'accuracy_no_death_cells']]
    param_cols = [c for c in df.columns if c not in ['mae_death_cells', 'f1_score', 'precision', 'recall']]
    # For every pair of hyperparameters, plot heatmaps for MAE and Accuracy
    for x_param, y_param in combinations(param_cols, 2):
        # Pivot tables for heatmaps (mean if multiple entries)
        mae_pivot = df.pivot_table(index=y_param, columns=x_param, values='mae_death_cells', aggfunc='mean')
        acc_pivot = df.pivot_table(index=y_param, columns=x_param, values='f1_score', aggfunc='mean')

        # Plot MAE heatmap (lower is better)
        plt.figure(figsize=(8, 6))
        height, width = mae_pivot.shape
        sns.heatmap(mae_pivot, annot=True, fmt=".3f", cmap='plasma_r', cbar_kws={'label': 'MAE Death Cells'}, 
                    # annot_kws={'size': max(6, 12 - 0.3 * max(height, width))}
                    )
        plt.title(f'MAE Death Cells: {y_param} vs {x_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.tight_layout()
        plt.savefig(save_dir / f'mae_{y_param}_vs_{x_param}.png')
        plt.close()

        # Plot No-Death Accuracy heatmap (higher is better)
        plt.figure(figsize=(8, 6))
        height, width = acc_pivot.shape
        sns.heatmap(acc_pivot, annot=True, fmt=".3f", cmap='plasma', cbar_kws={'label': 'F1-score No-Death Cells'}, 
                    # annot_kws={'size': max(6, 12 - 0.3 * max(height, width))}
                    )
        plt.title(f'F1-score No-Death Cells: {y_param} vs {x_param}')
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.tight_layout()
        plt.savefig(save_dir / f'accuracy_{y_param}_vs_{x_param}.png')
        plt.close()

    print(f"Heatmaps saved to {save_dir.resolve()}")
    
def plot_death_frame(death_frames_txt: Path, save_as: Path, hdf5_path: Path=SETTINGS.DATASET):
    true_all = np.genfromtxt(death_frames_txt, delimiter=',', usecols=1)
    # death_frames = death_frames.iloc[:, 1:-1]
    # death_frames.columns = ['Cell Idx', 'True']
    # death_frames = death_frames.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    with h5py.File(hdf5_path, 'r') as f:
        predicted_all = f['Cells']['Phase']['CellDeath'][0, :len(true_all)]
        
    # death_frames['Predicted'] = predicted

    valid_mask = ~np.isnan(true_all) & ~np.isnan(predicted_all)

    true = true_all[valid_mask]
    predicted = predicted_all[valid_mask]

    # == Plot ==
    plt.figure(figsize=(8, 6))
    plt.scatter(true, predicted, color='blue', label='Cell Death Prediction', marker='.')
    plt.plot([true.min(), true.max()],
            [true.min(), true.max()],
            color='red', linestyle='--', label='Perfect Prediction')

    plt.xlabel('True Death Frame')
    plt.ylabel('Predicted Death Frame')
    plt.title('Predicted vs. True Cell Death Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_as)
    
    
    # == Print avergae frame errors ==
    avg_error_abs = np.sum(np.abs(true - predicted)) / len(true)
    print(f"Average absolute error in cell death estimation is {avg_error_abs} frames.")
    
    avg_error = np.sum(predicted - true) / len(true)
    print(f"Average error in cell death estimation is {avg_error} frames.")
    
    delta_95 = np.percentile(np.abs(true-predicted), 95)
    print(f"95% of predictions are within ±{delta_95:.2f} frames of true time")
    
    delta_75 = np.percentile(np.abs(true-predicted), 75)
    print(f"75% of predictions are within ±{delta_75:.2f} frames of true time")
    
    

    # == Print classification metrics ==
    true_labels = ~np.isnan(true_all)           # True = death, False = no-death
    pred_labels = ~np.isnan(predicted_all)      # True = predicted death

    # Check if there are any labels at all to evaluate
    if np.any(true_labels) or np.any(~true_labels):
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 score:  {f1:.3f}\n")
    
def main():
    death_txt = '/home/ubuntu/PhagoPred/temp/03_10_deaths_new.txt'
    hyper_txt = '/home/ubuntu/PhagoPred/temp/death_hyperparamters.txt'

    # cell_death_hyperparameter_search(dataset=SETTINGS.DATASET, true_deaths_txt= death_txt, save_csv=True, csv_path = hyper_txt)
    # plot_hyperparam_heatmaps(hyper_txt, '/home/ubuntu/PhagoPred/temp/death_hyperparamters')
    plot_death_frame(death_txt, save_as='/home/ubuntu/PhagoPred/temp/death_error.png')
    
if __name__ == '__main__':
    main()
