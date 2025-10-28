from sksurv.tree import SurvivalTree
import numpy as np
from sksurv.metrics import concordance_index_censored
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from lifelines import KaplanMeierFitter
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
    
    n_slices_per_cell = 10  # adjust as needed
    X_train, y_train = dataset_to_xy(train_ds, n_slices=n_slices_per_cell)
    X_val, y_val = dataset_to_xy(val_ds, n_slices=n_slices_per_cell)
    
    print(len(X_train))
    
    survival_tree = SurvivalTree(max_depth=3)
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
    
    
    
    # train_nodes = survival_tree.apply(X_train.astype(np.float32))
    # n_nodes = survival_tree.tree_.node_count
    # node_samples = np.bincount(train_nodes, minlength=n_nodes)
    # node_events = np.bincount(train_nodes, weights=y_train["event"], minlength=n_nodes)
    # node_samples, node_events = get_samples_events(survival_tree, X_train, y_train)
    # print("Total samples:", np.sum(node_samples))   # should equal len(X_train)
    # print("Total events:", np.sum(node_events))     
    node_info = compute_node_survivals(survival_tree, X_train, y_train)
    plot_all_km_curves_single_plot(node_info, save_dir = save_as)
    
    
    flat_feature_names = [f"{feat}_{stat}" for feat in features for stat in train_ds.summary_funcs.keys()]
    
    plt.figure(figsize=(40, 20))  # width x height in inches
    plt_tree.plot_tree(survival_tree, feature_names=flat_feature_names, node_info=node_info, fontsize=4)
    plt.savefig(save_as / 'view_tree.png', dpi=500)  # higher DPI = sharper
    plt.close()
    # sklearn.tree.plot_tree(
    #     survival_tree,      # Use .tree_ for underlying tree
    #     feature_names=features,
    #     filled=True,
    #     rounded=True,
    #     fontsize=10
    # )

# def view_tree(model_path: Path):
#     with open(model_path / 'model.pickle', 'rb') as f:
#         model_data = pickle.load(f)
#     tree = model_data['tree']
#     features = model_data['features']

# def get_samples_events(survival_tree, X_train, y_train):
#     """Get the number of samples and number of events at each node of the tree."""
#     n_nodes = survival_tree.tree_.node_count
#     children_left = survival_tree.tree_.children_left
#     children_right = survival_tree.tree_.children_right
#     feature = survival_tree.tree_.feature
#     threshold = survival_tree.tree_.threshold

#     # Initialize counts
#     node_sample_counts = np.zeros(n_nodes, dtype=int)
#     node_event_counts = np.zeros(n_nodes, dtype=int)

#     # Assume y_train has two columns: [time, event] 
#     # event = 1 if event occurred, 0 if censored
#     # times, events = y_train[:, 0], y_train[:, 1]
#     events = y_train['event']

#     # Traverse each sample and increment counts along its path
#     for i, x in enumerate(X_train.astype(np.float32)):
#         node_id = 0  # start at root
#         node_sample_counts[node_id] += 1
#         node_event_counts[node_id] += events[i]
        
#         while children_left[node_id] != -1:  # while not a leaf
#             if x[feature[node_id]] <= threshold[node_id]:
#                 node_id = children_left[node_id]
#             else:
#                 node_id = children_right[node_id]
#             node_sample_counts[node_id] += 1
#             node_event_counts[node_id] += events[i]
#     return node_sample_counts, node_event_counts

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

        # # Percentiles
        # times = kmf.survival_function_.index.values
        # surv = kmf.survival_function_['KM_estimate'].values

        # for p in percentiles:
        #     threshold = 1 - p / 100
        #     idx = np.searchsorted(surv[::-1], threshold)
        #     node_percentiles[p][node_id] = times[::-1][idx] if idx < len(times) else np.inf
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