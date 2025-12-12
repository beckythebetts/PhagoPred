from sksurv.tree import SurvivalTree
import numpy as np
from sksurv.metrics import concordance_index_censored, brier_score
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from lifelines import KaplanMeierFitter
import sklearn
from tqdm import tqdm

from PhagoPred.survival_analysis.data import dataset
from PhagoPred.survival_analysis.models.decision_tree import plt_tree, eval


    
def train(train_paths: list, val_paths: list, features: list, save_as: Path, fixed_len:int = 10, max_time_to_death: int = None) -> None:
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
    
    # def dataset_to_xy(ds, n_slices=5):
    #     X, events, times = [], [], []
    #     for i in tqdm(range(len(ds)), desc=f'Processing dataset {ds}'):
    #         for _ in range(n_slices):
    #             features, _, event, time = ds[i]
    #             if features is not None:
    #                 X.append(features.flatten())
    #                 events.append(event)
    #                 times.append(time)
    #     X = np.vstack(X)
    #     y = np.array(list(zip(events, times)), dtype=[('event', '?'), ('time', '<f8')])
    #     return X, y
    
    n_slices_per_cell = 10  
    X_train, y_train = dataset.dataset_to_xy(train_ds, n_slices=n_slices_per_cell)
    print('Got train data')
    # X_val, y_val = dataset.dataset_to_xy(val_ds, n_slices=n_slices_per_cell)
    # print('Got val data')
    
    print(len(X_train))
    
    survival_tree = SurvivalTree(min_samples_leaf=1000, max_depth=5, max_leaf_nodes=3)
    survival_tree.fit(X_train, y_train)
    
    model_data = {
        "tree": survival_tree,
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
        
    # def c_index(y, pred):
    #     return concordance_index_censored(y["event"], y["time"], pred)[0]

    # pred_train = survival_tree.predict(X_train)
    # pred_val = survival_tree.predict(X_val)
    
    # print(y_train.shape ,y_val.shape, pred_train.shape, pred_val.shape)
    # print(f"C-index Train:     {c_index(y_train, pred_train):.3f}")
    # print(f"C-index Val:       {c_index(y_val, pred_val):.3f}")
    
    # # Choose evaluation time points
    # times_eval = np.arange(1, max_time_to_death, 1)

    # # Predict survival functions (list of StepFunction objects)
    # surv_funcs_train = survival_tree.predict_survival_function(X_train)
    # surv_funcs_val = survival_tree.predict_survival_function(X_val)

    # # Convert StepFunctions to numeric arrays
    # estimates_train = np.row_stack([fn(times_eval) for fn in surv_funcs_train])
    # estimates_val   = np.row_stack([fn(times_eval) for fn in surv_funcs_val])

    # # Compute Brier scores
    # bs_train = brier_score(y_train, y_train, estimates_train, times=times_eval)
    # bs_val   = brier_score(y_train, y_val,   estimates_val,   times=times_eval)

    # # Unpack times and scores
    # times_train, scores_train = bs_train
    # times_val, scores_val = bs_val

    # print("Train Brier Scores:")
    # for t, s in zip(times_train, scores_train):
    #     print(f"Time {t:.1f}: Brier Score {s:.4f}")

    # print("\nValidation Brier Scores:")
    # for t, s in zip(times_val, scores_val):
    #     print(f"Time {t:.1f}: Brier Score {s:.4f}")
    
    node_info = compute_node_survivals(survival_tree, X_train, y_train)
    plot_all_km_curves_single_plot(node_info, save_dir = save_as)
    
    
    flat_feature_names = [f"{feat}_{stat}" for feat in features for stat in train_ds.summary_funcs.keys()]
    
    plt.figure(figsize=(40, 20))  # width x height in inches
    plt_tree.plot_tree(survival_tree, feature_names=flat_feature_names, node_info=node_info, fontsize=4)
    plt.savefig(save_as / 'view_tree.png', dpi=500)  # higher DPI = sharper
    plt.close()


def compute_node_survivals(tree, X_train, y_train, percentiles=[5, 50, 95]):
    train_nodes = tree.apply(X_train.astype(np.float32))
    n_nodes = tree.tree_.node_count

    node_samples = np.zeros(n_nodes, dtype=int)
    node_events = np.zeros(n_nodes, dtype=int)
    node_indices = {i: [] for i in range(n_nodes)}
    node_median = np.full(n_nodes, np.nan)
    node_percentiles = {p: np.full(n_nodes, np.nan) for p in percentiles}
    node_kmf = {}

    for node_id in range(n_nodes):
        indices = np.where(train_nodes == node_id)[0]
        node_indices[node_id] = indices
        node_samples[node_id] = len(indices)
        if len(indices) == 0:
            continue

        y_subset = y_train[indices]
        node_events[node_id] = y_subset['event'].sum()

        # Compute KM curve for this node
        kmf = KaplanMeierFitter()
        kmf.fit(y_subset['time'], event_observed=y_subset['event'])
        node_kmf[node_id] = kmf

        # Median survival
        node_median[node_id] = kmf.median_survival_time_

        event_times = y_subset['time'][y_subset['event'] == 1]
        if len(event_times) > 0:
            for p in percentiles:
                node_percentiles[p][node_id] = np.percentile(event_times, p)
        else:
            for p in percentiles:
                node_percentiles[p][node_id] = np.nan  # no events in this node

    return {
        "samples": node_samples,
        "events": node_events,
        "indices": node_indices,
        "median": node_median,
        "percentiles": node_percentiles,
        "kmf": node_kmf
    }
    

def plot_all_km_curves_single_plot(node_info, save_dir: Path):
    node_kmf = node_info["kmf"]
    node_samples = node_info["samples"]
    node_events = node_info["events"]

    n_curves = len(node_kmf)
    cmap = plt.get_cmap('Set1')  # Choose colormap and number of discrete colors

    plt.figure(figsize=(12, 8))

    for i, (node_id, kmf) in enumerate(node_kmf.items()):
        color = cmap(i)
        # Plot survival function with confidence interval
        ax = kmf.plot_survival_function(ci_show=True, color=color, label=f"Node {node_id} (n={node_samples[node_id]}, e={node_events[node_id]})")

    plt.title("Kaplan-Meier Survival Curves for All Nodes")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_km_curves.png', dpi=300)
    
def main():
        # features = [
        #     'Area',
        #     # 'X',
        #     # 'Y',
        #     'Circularity',
        #     'Perimeter',
        #     'Displacement',
        #     'Speed',
        #     'Skeleton Length', 
        #     'Skeleton Branch Points', 
        #     'Skeleton End Points', 
        #     'Skeleton Branch Length Mean', 
        #     'Skeleton Branch Length Std',
        #     'Skeleton Branch Length Max',
        #     # 'UMAP 1',
        #     # 'UMAP 2',
        #     # 'Mode 0',
        #     # 'Mode 1',
        #     # 'Mode 2',
        #     # 'Mode 3',
        #     # 'Mode 4',
        #     # 'Speed',
        #     'Phagocytes within 100 pixels',
        #     'Phagocytes within 250 pixels',
        #     'Phagocytes within 500 pixels',
        #     # 'Total Fluorescence', 
        #     # 'Fluorescence Distance Mean', 
        #     # 'Fluorescence Distance Variance',
        #     ] 
        features = [
        '0',
        '1',
        '2',
        '3',
        ] 
    
        
        # train_hdf5_paths=[
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
            
        # ]
        # val_hdf5_paths=[
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '07_10_0.h5',
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '28_10_2500.h5',
        #     Path('PhagoPred') / 'Datasets' / 'ExposureTest' / '10_10_5000.h5',
        # ]
        train_datasets = [Path('PhagoPred')/'Datasets'/'synthetic.h5']
        val_datasets = [Path('PhagoPred')/'Datasets'/'val_synthetic.h5']
        train(train_datasets, val_datasets, features, fixed_len=100, save_as=Path('PhagoPred') / 'survival_analysis' / 'models' / 'decision_tree' / 'test', max_time_to_death=100)
        
if __name__ == '__main__':
    main()