import sklearn
import scipy
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm



from PhagoPred.prediction.datasets import SummaryStatsCellDataset

def get_summary_stats_arrays(dataset: SummaryStatsCellDataset):
    """
    Get summary statistics arrays for all cells in the dataset.

    Args:
        dataset (SummaryStatsCellDataset): The dataset containing cell data.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - summary_stats: Array of summary statistics for each cell.
            - alive: Array of booleans indicating if each cell is alive.
    """
    summary_stats = []
    alive = []
    print('\n=== Loading Dataset ===\n')
    dataset = dataset[list(range(len(dataset)))]
    for stats, is_alive in dataset:
        summary_stats.append(stats)
        alive.append(is_alive)
    
    return np.array(summary_stats), np.array(alive)


    # for i in tqdm(range(len(dataset))):
    #     stats, is_alive = dataset[i]
    #     if stats is not None:
    #         summary_stats.append(stats.flatten())
    #         alive.append(is_alive)

    # return np.array(summary_stats), np.array(alive)

def train_eval(
        # hdf5_files: list[Path],
        dataset: SummaryStatsCellDataset = None,
        ):
    """
    Train and evaluate a decision tree model using summary statistics from the dataset.
    """

    # dataset = SummaryStatsCellDataset(hdf5_files,
    #                                   fixed_length=20,
    #                                   pre_death_frames=0,
    #                                   include_alive=True,
    #                                   num_alive_samples=1,
    #                                   mode='train',
    #                                   )
    
    X, y = get_summary_stats_arrays(dataset)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)

    # Create and train the decision tree classifier
    clf = sklearn.tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Save the model
    joblib.dump(clf, Path(__file__).parent / 'decision_tree_model.pkl')

    # Evaluate the model
    accuracy = clf.score(X_test, y_test)
    classification_report = sklearn.metrics.classification_report(
        y_test, 
        clf.predict(X_test),
        target_names=["Dead", "Alive"])

    print(f"Model accuracy: {accuracy:.2f}")
    with open(Path(__file__).parent / 'decision_tree_accuracy.txt', 'w') as f:
        f.write(f"Model accuracy: {accuracy:.2f}\n{classification_report}")
    save_decision_tree_plot(clf, feature_names=dataset.feature_names)

def inference(hdf5_files: list[Path]):
    """
    Perform inference using the trained decision tree model on the dataset.

    Args:
        hdf5_files (list[Path]): List of paths to HDF5 files containing the dataset.
    """
    dataset = SummaryStatsCellDataset(hdf5_files,
                                      fixed_length=20,
                                      pre_death_frames=0,
                                      include_alive=True,
                                      num_alive_samples=1,
                                      )
    X, _ = get_summary_stats_arrays(dataset)

    # Load the trained model
    clf = joblib.load(Path(__file__).parent / 'decision_tree_model.pkl')

    # Perform inference
    predictions = clf.predict(X)
    print(f"Predictions: {predictions}")

def save_decision_tree_plot(clf=None, feature_names=None, save_path=Path(__file__).parent / 'decision_tree.png', max_depth=4):
    if clf is None:
        clf = joblib.load(Path(__file__).parent / 'decision_tree_model.pkl')
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        filled=True,
        feature_names=feature_names,
        class_names=["Dead", "Alive"],
        max_depth=max_depth,
        fontsize=15
    )
    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n✅ Decision tree plot saved to {save_path}")

def main():
    hdf5_files = [Path('PhagoPred') / 'Datasets' / '24_06.h5']
    dataset = SummaryStatsCellDataset(hdf5_files,
                                      fixed_length=20,
                                      pre_death_frames=0,
                                      include_alive=True,
                                      num_alive_samples=1,
                                      )

    train_eval(dataset)
    save_decision_tree_plot(feature_names=dataset.feature_names)

if __name__ == '__main__':

    main()
