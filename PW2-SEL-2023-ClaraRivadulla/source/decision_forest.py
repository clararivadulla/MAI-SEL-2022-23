import numpy as np
from node import Node
from collections import Counter


def most_common_class(y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]


class DecisionTree:

    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def gini(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        return 1 - sum([((np.sum(y == c)) / n_samples) ** 2 for c in classes])

    def predict(self, X):
        return [self.predict_x(x) for x in X]

    def predict_x(self, x):
        node = self.tree
        while node.left:
            if isinstance(x[node.feature_idx], str):
                if x[node.feature_idx] == node.threshold:
                    node = node.left
                else:
                    node = node.right
            else:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return node.pred_class

    def grow_tree(self, X, y, depth=0):
        node = Node(pred_class=most_common_class(y))
        if depth < self.max_depth:
            feature_idx, threshold = self.best_split(X, y)
            if feature_idx is not None:
                if isinstance(X[0, feature_idx], str):
                    left_idxs = np.where(X[:, feature_idx] == threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] != threshold)[0]
                else:
                    left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                node.feature_idx = feature_idx
                node.threshold = threshold
                node.left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
                node.right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return node

    def best_split(self, X, y):
        m = len(y)
        if m <= 1:
            return None, None
        parent = [np.sum(y == c) for c in self.classes]
        best_gini = 1.0 - sum((n / m) ** 2 for n in parent)
        best_threshold = None
        best_idx = None
        n_samples = X.shape[0]

        # Loop over every feature j and split every pred_class it takes
        for feature_idx in range(self.features):
            # The midpoint between each pair of sorted adjacent pred_classs is taken as a possible split-point
            thresholds = np.unique(X[:, feature_idx])
            if not isinstance(X[0, feature_idx], str):
                if len(thresholds) > 1:
                    thresholds = (thresholds[:-1] + thresholds[1:]) / 2
            for threshold in thresholds:
                if isinstance(X[0, feature_idx], str):
                    left_idxs = np.where(X[:, feature_idx] == threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] != threshold)[0]
                else:
                    left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                # print(left_idxs, right_idxs)
                gini = (len(left_idxs) / n_samples) * self.gini(y[left_idxs]) + (
                        len(right_idxs) / n_samples) * self.gini(y[right_idxs])
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_threshold = threshold
        # print(best_gini, best_idx, best_threshold)
        return best_idx, best_threshold


class DecisionForest:
    def __init__(self, max_depth=2, min_impurity=1e-7, NT=2, F=2):
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.NT = NT
        self.F = F
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for i in range(self.NT):
            features = np.random.choice(n_features, size=self.F, replace=False)
            X_subset = X[:, features]
            # print("SUBSET:" + str(X_subset))
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y)
            self.trees.append((tree, features))

    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            # print(tree, features)
            X_subset = X[:, features]
            # print("SUBSET TO PRED:" + str(X_subset))
            prediction = tree.predict(X_subset)
            # print("PREDICTION:" + str(prediction))
            predictions.append(prediction)
        y_pred = []
        for i in range(len(predictions[0])):
            counter = Counter([p[i] for p in predictions])
            y_pred.append(counter.most_common(1)[0][0])
        return y_pred
