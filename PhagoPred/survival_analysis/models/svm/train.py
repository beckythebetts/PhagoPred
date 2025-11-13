from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from lifelines import KaplanMeierFitter
import sklearn
from tqdm import tqdm

from PhagoPred.survival_analysis.data import dataset
from PhagoPred.survival_analysis.models.svm import eval
from PhagoPred.utils import tools


    
def train(train_paths: list, val_paths: list, features: list, save_as: Path, fixed_len:int = 10, max_time_to_death: int = None) -> None:
    
    tools.remake_dir(save_as)
    
    train_ds = dataset.CellDataset(
        train_paths,
        features=features,
        summary_stats=True,
        fixed_len=fixed_len,
        max_time_to_death=max_time_to_death
    )
    
    means, stds = train_ds.get_normalization_stats()
    
    val_ds = dataset.CellDataset(
        val_paths,
        features=features,
        means=means,
        stds=stds,
        summary_stats=True,
        fixed_len=fixed_len,
        max_time_to_death=max_time_to_death
    )
    
    n_slices_per_cell = 10  
    X_train, y_train = dataset.dataset_to_xy(train_ds, n_slices=n_slices_per_cell)
    
    # assert ~np.any(np.isnan(X_train))
    # assert ~np.any(np.isnan(y_train['time']))
    # assert ~np.any(np.isnan(y_train['event']))
    
    svm = FastSurvivalSVM(rank_ratio=0.5)
    svm.fit(X_train, y_train)
    
    # survival_tree = SurvivalTree(min_samples_leaf=1000, max_depth=5, max_leaf_nodes=5)
    # survival_tree.fit(X_train, y_train)
    
    model_data = {
        "svm": svm,
        "y_train": y_train,
        "dataset_args" : {
            "means": means,
            "stds": stds,
            "features": features,
            "fixed_len": fixed_len,
            "max_time_to_death": max_time_to_death,
        }
    }
    
    with open(save_as / 'model.pickle', 'wb') as f:
        pickle.dump(model_data, f)

    for ds, name in zip((train_ds, val_ds), ('train', 'val')):
        eval.eval(model_data, ds, save_as / name)

def main():
        features = [
            'Area',
            # 'X',
            # 'Y',
            'Circularity',
            'Perimeter',
            'Displacement',
            'Speed',
            'Skeleton Length', 
            'Skeleton Branch Points', 
            'Skeleton End Points', 
            'Skeleton Branch Length Mean', 
            'Skeleton Branch Length Std',
            'Skeleton Branch Length Max',
            # 'UMAP 1',
            # 'UMAP 2',
            # 'Mode 0',
            # 'Mode 1',
            # 'Mode 2',
            # 'Mode 3',
            # 'Mode 4',
            # 'Speed',
            'Phagocytes within 100 pixels',
            'Phagocytes within 250 pixels',
            'Phagocytes within 500 pixels',
            # 'Total Fluorescence', 
            # 'Fluorescence Distance Mean', 
            # 'Fluorescence Distance Variance',
            ] 
        
        train_hdf5_paths=[
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
            
        ]
        val_hdf5_paths=[
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
        ]
        
        train(train_hdf5_paths, val_hdf5_paths, features, fixed_len=100, save_as=Path('PhagoPred') / 'survival_analysis' / 'models' / 'svm' / 'test', max_time_to_death=100)
        
if __name__ == '__main__':
    main()