import sklearn
import scipy
import numpy as np

class DecisionTree:

    def __init__(self, train_data, train_labels, test_data, test_labels):
        """
        
        """
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.classifier = sklearn.tree.DecisionTreeClassifier()

    def train(self):
        self.classifier.fit(self.train_data, self.train_labels)

    def predict(self):
        pred_labels = self.classifier.predict(self.test_data)

    def draw_tree(self):
        sklearn.tree.plot_tree(self.classifier)

def get_summary_stats(raw_data):
    """
    Args:
        raw_data [num_sequences, sequence_len, raw_features]
    """
    means = np.mean(raw_data, axis=1)
    stds = np.std(raw_data, axis=1)
    skew = scipy.stats.skew(raw_data, axis=1)
    differences = np.diff(raw_data, n=1, axis=1)
    tot_ascent = np.sum(differences[differences>0], axis=1)
    tot_descent = np.sum(differences[differences<0], axis=1)
    return np.concatenate((means, stds, skew, tot_ascent, tot_descent), axis=1)


