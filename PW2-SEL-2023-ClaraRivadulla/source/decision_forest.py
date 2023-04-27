import numpy as np
from node import Node

class DecisionTree:
    def __init__(self, max_depth=2, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def gini(self, y):
        n_samples = len(y)
        classes = np.unique(y)
        return 1 - sum([((np.sum(y == c)) / n_samples) ** 2 for c in classes])

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or self.gini(y) < self.min_impurity:
            print(y)
            most_common_value = np.unique(y, return_counts=True)[0][np.unique(y, return_counts=True)[1].argmax()]
            print(most_common_value)
            return Node(value=most_common_value) # Return the most common class

        if len(np.unique(y)) == 1:
            return Node(None, None, None, y[0])

        if n_features == 0:
            return Node(None, None, None, max(set(y), key=y.count))

        best_idx, best_threshold = self.split(X, y, n_samples, n_features)

        if isinstance(X[0, best_idx], str):
            left_idxs = np.where(X[:, best_idx] == best_threshold)[0]
            right_idxs = np.where(X[:, best_idx] != best_threshold)[0]
        else:
            left_idxs = np.where(X[:, best_idx] <= best_threshold)[0]
            right_idxs = np.where(X[:, best_idx] > best_threshold)[0]

        left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature_idx=best_idx, left=left, right=right, threshold=best_threshold)

    def predict(self, X):
        return [self.predict_x(x) for x in X]

    def predict_x(self, x):
        node = self.root
        while node.left:
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def split(self, X, y, n_samples, n_features):

        min_gini = np.inf
        best_threshold = None
        best_idx = None

        # Loop over every feature j and split every value it takes
        for feature_idx in range(n_features):
            # The midpoint between each pair of sorted adjacent values is taken as a possible split-point
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
                #print(left_idxs, right_idxs)
                gini = (len(left_idxs) / n_samples) * self.gini(y[left_idxs]) + (
                            len(right_idxs) / n_samples) * self.gini(y[right_idxs])
                if gini < min_gini:
                    min_gini = gini
                    best_idx = feature_idx
                    best_threshold = threshold
        print(min_gini, best_idx, best_threshold)
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
        print(n_samples, n_features)
        for i in range(self.NT):
            print(n_features)
            features = np.random.choice(n_features, size=self.F)
            print(features)
            X_subset = X[:, features]
            print(X)
            print(X_subset)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y)
            self.trees.append((tree, features))

    def predict(self, X):
        predictions = []
        for tree, features in self.trees:
            data_subset = [X[j] for j in features]
            prediction = tree.classify(data_subset, tree.tree)
            predictions.append(prediction)
        return max(set(predictions), key=predictions.count)
