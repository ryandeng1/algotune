import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    def solve(self, problem, **kwargs) -> sparse.spmatrix:
        """Compute the matrix exponential of a sparse CSC matrix."""
        A = problem["matrix"]
        return expm(A)
