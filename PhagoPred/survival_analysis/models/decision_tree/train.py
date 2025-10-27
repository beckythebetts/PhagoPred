from sksurv.tree import SurvivalTree
import numpy as np
from sksurv.metrics import concordance_index_censored
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import sklearn

from PhagoPred.survival_analysis.data import dataset
from PhagoPred.survival_analysis.models.decision_tree import plt_tree


def train(train_paths: list, val_paths: list, features: list, save_as: Path, fixed_len:int = 10, ) -> None:
    train_ds = dataset.CellDataset(
        train_paths,
        features=features,
        summary_stats=True,
        fixed_len=fixed_len
    )
    
    means, stds = train_ds.get_normalization_stats()
    
    val_ds = dataset.CellDataset(
        val_paths,
        features=features,
        means=means,
        stds=stds,
        summary_stats=True,
        fixed_len=fixed_len,
    )
    
    def dataset_to_xy(ds, n_slices=5):
        X, events, times = [], [], []
        for i in range(len(ds)):
            for _ in range(n_slices):
                features, _, event, time = ds[i]
                if features is not None:
                    X.append(features.flatten())
                    events.append(event)
                    times.append(time)
        X = np.vstack(X)
        y = np.array(list(zip(events, times)), dtype=[('event', '?'), ('time', '<f8')])
        return X, y
    
    n_slices_per_cell = 5  # adjust as needed
    X_train, y_train = dataset_to_xy(train_ds, n_slices=n_slices_per_cell)
    X_val, y_val = dataset_to_xy(val_ds, n_slices=n_slices_per_cell)
    
    print(len(X_train))
    
    survival_tree = SurvivalTree()
    survival_tree.fit(X_train, y_train)
    
    model_data = {
        "tree": survival_tree,
        "means": means,
        "stds": stds,
        "features": features,  # optional but useful
    }
    
    with open(save_as / 'model.pickle', 'wb') as f:
        pickle.dump(model_data, f)

    
    
    def c_index(y, pred):
        return concordance_index_censored(y["event"], y["time"], pred)[0]

    pred_train = survival_tree.predict(X_train)
    pred_val = survival_tree.predict(X_val)

    print(f"C-index Train: {c_index(y_train, pred_train):.3f}")
    print(f"C-index Val:   {c_index(y_val, pred_val):.3f}")
    
    flat_feature_names = [f"{feat}_{stat}" for feat in features for stat in train_ds.summary_funcs.keys()]
    
    plt_tree.plot_tree(survival_tree, feature_names=flat_feature_names)
    plt.savefig(save_as / 'view_tree.png')
    # sklearn.tree.plot_tree(
    #     survival_tree,      # Use .tree_ for underlying tree
    #     feature_names=features,
    #     filled=True,
    #     rounded=True,
    #     fontsize=10
    # )

    
    
    
def main():
        features = [
            'Area',
            # 'X',
            # 'Y',
            'Circularity',
            'Perimeter',
            'Displacement',
            'Skeleton Length', 
            'Skeleton Branch Points', 
            'Skeleton End Points', 
            'Skeleton Branch Length Mean', 
            'Skeleton Branch Length Std',
            'Skeleton Branch Length Max',
            'UMAP 1',
            'UMAP 2',
            # 'Mode 0',
            # 'Mode 1',
            # 'Mode 2',
            # 'Mode 3',
            # 'Mode 4',
            # 'Speed',
            'Phagocytes within 100 pixels',
            'Phagocytes within 250 pixels',
            'Phagocytes within 500 pixels',
            'Total Fluorescence', 
            'Fluorescence Distance Mean', 
            'Fluorescence Distance Variance',
            ] 
        
        train_hdf5_paths=[
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
            
        ]
        val_hdf5_paths=[
            # Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
            Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
        ]
        
        train(train_hdf5_paths, val_hdf5_paths, features, fixed_len=100, save_as=Path('PhagoPred') / 'survival_analysis' / 'models' / 'decision_tree' / 'test')
        
if __name__ == '__main__':
    main()