import numpy as np
from node import Node

class CART:
    def __init__(self, max_depth=2, min_samples_split=2, F=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.F = F

    def fit(self, X, y):
        #print(X)
        self.n_classes = len(np.unique(y))
        self.tree = self.grow_tree(X, y)

    def gini(self, y):
        n_samples = len(y)
        return 1 - sum([(np.sum(y == c) / n_samples) ** 2 for c in range(self.n_classes)])

    def grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        #print(n_features)
        #print(n_samples)
        #print(X)

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            y = y.astype(int)
            return Node(value=np.bincount(y).argmax()) # Return the most common class

        min_gini = np.inf
        best_j = 0
        best_split = np.unique(X[:, best_j])[0]
        features_idx = np.random.choice(n_features, size=self.F, replace=False)
        #print(features_idx)
        # Loop over every feature j and split every value it takes
        for j in features_idx:
            # The midpoint between each pair of sorted adjacent values is taken as a possible split-point
            splits = np.unique(X[:, j])
            #print("SPLITS: " + str(splits))
            if not isinstance(X[0, j], str):
                if len(splits) > 1:
                    splits = (splits[:-1] + splits[1:]) / 2
            #print("SPLITS MIDTERRM: " + str(splits))
            for split in splits:
                if isinstance(X[0, j], str):
                    left_idxs = np.where(X[:, j] == split)[0]
                    right_idxs = np.where(X[:, j] != split)[0]
                else:
                    left_idxs = np.where(X[:, j] <= split)[0]
                    right_idxs = np.where(X[:, j] > split)[0]
                #print("len left: " + str(len(left_idxs)))
                #print("len right: " + str(len(right_idxs)))
                #print("gini left: " + str(self.gini(y[left_idxs])))
                #print("gini right: " + str(self.gini(y[right_idxs])))
                if len(right_idxs) == 0 or len(left_idxs) == 0:
                    y = y.astype(int)
                    return Node(value=np.bincount(y).argmax())
                else:
                    gini = (len(left_idxs)/n_samples)*self.gini(y[left_idxs]) + (len(right_idxs)/n_samples)*self.gini(y[right_idxs])
                #print("gini: " + str(gini))
                if gini < min_gini:
                    min_gini = gini
                    best_j = j
                    best_split = split

        #print(min_gini, best_j, best_split)
        if isinstance(X[0, j], str):
            left_idxs = np.where(X[:, best_j] == best_split)[0]
            right_idxs = np.where(X[:, best_j] != best_split)[0]
        else:
            left_idxs = np.where(X[:, best_j] <= best_split)[0]
            right_idxs = np.where(X[:, best_j] > best_split)[0]

        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(j=best_j, left=left, right=right, split=best_split)

    def predict(self, X):
        return [self.predict_x(x) for x in X]

    def predict_x(self, x):
        node = self.tree
        while node.left:
            if x[node.j] <= node.split:
                node = node.left
            else:
                node = node.right
        return node.value

class RF:
    def __init__(self, F=2, NT=2, max_depth=2):
        self.F = F
        self.NT = NT
        self.trees = []
        self.max_depth = max_depth

    def fit(self, X, y):
        for i in range(self.NT):
            tree = CART(max_depth=self.max_depth, F=self.F)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.NT))
        i = 0
        for tree in self.trees:
            predictions[:, i] = tree.predict(X)
            i += 1
        predictions = predictions.astype(int)
        y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)
        return y_pred
