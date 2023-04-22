class RF:
    def __init__(self, F=2, NT=2, max_depth=2):
        self.F = F
        self.NT = NT
        self.trees = []
        self.max_depth = max_depth