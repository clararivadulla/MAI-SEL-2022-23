import numpy as np
from node import Node
from collections import Counter


def most_common_class(y):
    """ Function to get the most common class in a subset or set of classes 'y' """
    counter = Counter(y)
    return counter.most_common(1)[0][0]


def feature_importances(tree):
    """ Function that computes the importance of the features by traversing the tree """
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

    def __init__(self, max_depth=100, min_impurity=1e-7):
        self.max_depth = max_depth
        self.min_impurity = min_impurity

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y)

    def gini(self, y):
        """ This function computes the gini impurity for a given set or subset of classes 'y' """
        n_samples = len(y)
        classes = np.unique(y)
        return 1 - sum([((np.sum(y == c)) / n_samples) ** 2 for c in classes])

    def predict(self, X):
        """ Predict the class of every data instance x in X """
        return [self.predict_x(x) for x in X]

    def predict_x(self, x):
        """ Start from the root node and keep following left or right
        depending on where the value of the best feature falls until a leaf node is reached.
         Then, return the predicted class """
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
        node = Node(pred_class=most_common_class(y)) # Create a node with the most common class as the predicted class
        if depth < self.max_depth: # If maximum depth has not been reached yet
            feature_idx, threshold, impurity = self.best_split(X, y) # Find the best possible split and keep the index of the feature. the threshold and the impurity
            if feature_idx is not None:
                if isinstance(X[0, feature_idx], str):
                    left_idxs = np.where(X[:, feature_idx] == threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] != threshold)[0]
                else:
                    left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                    right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                node.feature_idx = feature_idx # Save the node's chosen feature index
                node.threshold = threshold # Save the node's chosen threshold
                node.impurity = impurity # Save the node's impurity
                node.left = self.grow_tree(X[left_idxs], y[left_idxs], depth + 1) # Grow the tree to the left
                node.right = self.grow_tree(X[right_idxs], y[right_idxs], depth + 1) # Grow the tree to the right
        return node # Return the node

    def best_split(self, X, y):
        len_y = len(y)
        if len_y <= 1: # If there's just one class left, return None (as a leaf node has been reached)
            return None, None, None
        parent = [np.sum(y == c) for c in self.classes]
        best_gini = 1.0 - sum((n / len_y) ** 2 for n in parent) # Calculate the gini impurity of the parent node as the temporary best
        best_threshold = None
        best_idx = None
        n_samples = X.shape[0]

        if best_gini >= self.min_impurity:
            # Loop over every feature j
            for feature_idx in range(self.features):
                # The midpoint between each pair of sorted adjacent pred_class is taken as a possible split-point (only if the feature is numeric)
                thresholds = np.unique(X[:, feature_idx])
                if not isinstance(X[0, feature_idx], str):
                    if len(thresholds) > 1:
                        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
                """ For every possible threshold, calculate the gini impurity and keep 
                the best values of gini, the feature index and the threshold """
                for threshold in thresholds:
                    if isinstance(X[0, feature_idx], str):
                        left_idxs = np.where(X[:, feature_idx] == threshold)[0]
                        right_idxs = np.where(X[:, feature_idx] != threshold)[0]
                    else:
                        left_idxs = np.where(X[:, feature_idx] <= threshold)[0]
                        right_idxs = np.where(X[:, feature_idx] > threshold)[0]
                    gini = (len(left_idxs) / n_samples) * self.gini(y[left_idxs]) + (
                            len(right_idxs) / n_samples) * self.gini(y[right_idxs]) # Compute the gini impurity
                    if gini < best_gini: # If the gini impurity has improved, keep these values
                        best_gini = gini
                        best_idx = feature_idx
                        best_threshold = threshold
        return best_idx, best_threshold, best_gini # Return the best values


class DecisionForest:
    def __init__(self, max_depth=100, min_impurity=1e-7, NT=2, F=2, feature_names=None):
        self.max_depth = max_depth
        self.min_impurity = min_impurity
        self.NT = NT
        self.F = F
        self.trees = []
        self.feature_names = feature_names
        self.overall_feature_importances = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for i in self.feature_names:
            self.overall_feature_importances[i] = 0
        # Create NT trees and fit each one of them with a subset of X with F random features
        for i in range(self.NT):
            # Pick F random features
            features = np.random.choice(n_features, size=self.F, replace=False)
            features_n = self.feature_names[features]
            X_subset = X[:, features]
            tree = DecisionTree(max_depth=self.max_depth, min_impurity=self.min_impurity)
            tree.fit(X_subset, y) # Fit the tree with a subset of X (taking into consideration the F random features chosen
            importances = feature_importances(tree) # Compute the importance of the tree's features
            # Update the overall feature importances
            i = 0
            for feature in features_n:
                self.overall_feature_importances[feature] += importances[i]
                i += 1
            self.trees.append((tree, features))

    def predict(self, X):
        """ Function that gets the predictions of all the trees of the Decision Trees
        and returns the predictions that appear more often """
        predictions = []
        for tree, features in self.trees:
            X_subset = X[:, features]
            prediction = tree.predict(X_subset)
            predictions.append(prediction)
        y_pred = []
        for i in range(len(predictions[0])):
            counter = Counter([p[i] for p in predictions])
            y_pred.append(counter.most_common(1)[0][0])
        return y_pred

    def print_most_important_features(self):
        """ Function to print the 3 most important features of the model. """
        sorted_items = sorted(self.overall_feature_importances.items(), key=lambda x: x[1], reverse=True)
        for i, (key, value) in enumerate(sorted_items[:3], 1):
            print(f"{i}. {key}")
