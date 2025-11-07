from pathlib import Path

from sksurv.metrics import concordance_index_censored, brier_score, integrated_brier_score
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from PhagoPred.survival_analysis.data import dataset
from PhagoPred.survival_analysis.models.decision_tree import train
from PhagoPred.utils import tools

def eval(model_data: dict, ds: list, save_to: Path) -> None:
    """Evaulate a survival decsion tree."""
    
    tools.remake_dir(save_to)
    
    if isinstance(ds, list):
        ds = dataset.CellDataset(
            ds,
            summary_stats=True,
            **model_data['dataset_args'],
            
        )
    
    X, y = dataset.dataset_to_xy(ds, n_slices=10)
    
    survival_tree = model_data['tree']
    
    pred_risk = survival_tree.predict(X)
    surv_funcs = survival_tree.predict_survival_function(X)
    eval_times = np.unique(model_data['y_train']['time'])
    eval_times = eval_times[(eval_times < y['time'].max())]
    surv_funcs_array = np.row_stack([fn(eval_times) for fn in surv_funcs])
    
    c_index = concordance_index_censored(y['event'], y['time'], pred_risk)[0]
    eval_times, brier_scores = brier_score(model_data['y_train'], y, surv_funcs_array, times=eval_times)
    integrated_brier = integrated_brier_score(model_data['y_train'], y, surv_funcs_array, times=eval_times)
    
    print(eval_times, brier_scores)
    scores_json = {
        'c_index': c_index, 
        'intagrated_brier': integrated_brier,
        'brier_scores': {time: score for time, score in zip(eval_times, brier_scores)}
        }
    
    with open(save_to / 'scores.json', 'w') as f:
        json.dump(scores_json, f)
        
    plot_brier_scores(eval_times, brier_scores, save_to / 'brier_scores.png')
        
        
def plot_brier_scores(times: list, scores: list, save_as: Path) -> None:
    """Plot Brier score over evaluation times and save as a PNG."""
    plt.figure(figsize=(6, 4))
    plt.plot(times, scores, marker='o', linestyle='-', color='tab:blue', label='Brier score')
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.title("Brier Score over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
    
    
def main():
    model_path = Path('/home/ubuntu/PhagoPred/PhagoPred/survival_analysis/models/decision_tree/test/model.pickle')
    val_paths = [
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
    ]
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    eval(model_data, val_paths, save_to=model_path.parent)    
    
if __name__ == '__main__':
    main()
    