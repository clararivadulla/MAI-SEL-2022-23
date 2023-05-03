import numpy as np
from node import Node
from collections import Counter
from sklearn.utils import resample


def most_common_class(y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]


def feature_importances(tree):
    importances = np.zeros(tree.features)

    def traverse_tree(node, importance):
        if node.feature_idx is not None:
            importances[node.feature_idx] += importance
            traverse_tree(node.left, importance * (1 - node.impurity))
            traverse_tree(node.right, importance * (1 - node.impurity))

    traverse_tree(tree.root, 1)
    importances /= np.sum(importances)
    return importances


class DecisionTree:

    def __init__(self, max_depth=100, min_impurity=1e-7, F=2):
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.F = F

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y)

    def gini(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        return 1 - sum([((np.sum(y == c)) / n_samples) ** 2 for c in classes])

    def predict(self, X):
        return [self.predict_x(x) for x in X]

    def predict_x(self, x):
        node = self.root
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
            feature_idx, threshold, impurity = self.best_split(X, y)
            if feature_idx is not None:
                if isinstance(X[0, feature_idx], str):
                    left_idxs = np.where(X[:, feature_idx] == threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] != threshold)[0]
                else:
                    left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                node.feature_idx = feature_idx
                node.threshold = threshold
                node.impurity = impurity
                node.left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
                node.right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return node

    def best_split(self, X, y):
        n_samples, n_features = X.shape
        m = len(y)
        if m <= 1:
            return None, None, None
        parent = [np.sum(y == c) for c in self.classes]
        best_gini = 1.0 - sum((n / m) ** 2 for n in parent)
        best_threshold = None
        best_idx = None

        features_idxs = np.random.choice(n_features, size=self.F, replace=False)
        if best_gini >= self.min_impurity:
            # Loop over every feature j and split every pred_class it takes
            for feature_idx in features_idxs:
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
                    gini = (len(left_idxs) / n_samples) * self.gini(y[left_idxs]) + (
                            len(right_idxs) / n_samples) * self.gini(y[right_idxs])
                    if gini < best_gini:
                        best_gini = gini
                        best_idx = feature_idx
                        best_threshold = threshold
        return best_idx, best_threshold, best_gini


class RandomForest:
    def __init__(self, max_depth=100, min_impurity=1e-7, NT=2, F=2, feature_names=None):
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.NT = NT
        self.F = F
        self.trees = []
        self.feature_names = feature_names
        self.overall_feature_importances = {}

    def fit(self, X, y):
        for i in self.feature_names:
            self.overall_feature_importances[i] = 0
        for i in range(self.NT):
            random_indices = np.random.randint(len(X), size=len(X))
            bootstrap_X = X[random_indices]
            bootstrap_y = y[random_indices]
            tree = DecisionTree(max_depth=self.max_depth, F=self.F)
            tree.fit(bootstrap_X, bootstrap_y)
            importances = feature_importances(tree)
            i = 0
            for feature in self.feature_names:
                self.overall_feature_importances[feature] = importances[i]
                i += 1
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        y_pred = []
        for i in range(len(predictions[0])):
            counter = Counter([p[i] for p in predictions])
            y_pred.append(counter.most_common(1)[0][0])
        return y_pred

    def print_most_important_features(self):
        sorted_items = sorted(self.overall_feature_importances.items(), key=lambda x: x[1], reverse=True)
        for i, (key, value) in enumerate(sorted_items[:3], 1):
            print(f"{i}. {key}")
