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
    
    svm = model_data['svm']
    
    pred_times = svm.predict(X)
    # # outputs are log transformed times
    # pred_times = np.exp(pred_times)
    print(np.unique(pred_times))
    
    plot_pred_vs_true_times(np.array(pred_times), np.array(y['time']), np.array(y['event']), save_as=save_to / 'results.png')

        
        
def plot_pred_vs_true_times(pred_times: np.ndarray, true_times: np.ndarray, events: np.ndarray, save_as: Path) -> None:
    """Plot Brier score over evaluation times and save as a PNG."""
    plt.figure(figsize=(6, 4))
    
    pred_times = pred_times[events]
    true_times = true_times[events]
    
    plt.scatter(true_times, pred_times, marker='o', color='tab:blue')
    plt.xlabel("True surivival time")
    plt.ylabel("Predicted survival time")
    plt.ylim(0, np.max(true_times)*1.1)
    plt.title("Predicted survival times")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
    
    
def main():
    # model_path = Path('/home/ubuntu/PhagoPred/PhagoPred/survival_analysis/models/svm/test/model.pickle')
    model_path = Path(r'PhagoPred\survival_analysis\models\svm\test\model.pickle')
    val_paths = [
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
    ]
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    eval(model_data, val_paths, save_to=model_path.parent / 'val')    
    
if __name__ == "__main__":
    main()