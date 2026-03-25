import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Return the outer product of two vectors."""
        vec1, vec2 = problem
        return np.outer(vec1, vec2)
