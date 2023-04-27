class Node:
    def __init__(self, feature_idx=None, left=None, right=None, threshold=None, value=None):
        self.feature_idx = feature_idx
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value
