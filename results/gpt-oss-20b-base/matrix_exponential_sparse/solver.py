import numpy as np
from scipy import sparse
from scipy.linalg import expm as lec_expm

class Solver:
    """
    Solver that computes the matrix exponential of a sparse matrix.
    For small matrices a dense implementation is used for speed,
    otherwise the scipy sparse implementation is employed.
    """
    # Empirical threshold: matrices with <= 5000 elements are treated as small
    _SMALL_MATRIX_LIMIT = 5000

    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        A = problem["matrix"]
        # Ensure matrix is square
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for exponential")
        # Small matrices: convert to dense, use scipy.linalg.expm, convert back
        if int(A.shape[0] * A.shape[1]) <= self._SMALL_MATRIX_LIMIT:
            dense = A.toarray()
            exp_dense = lec_expm(dense)
            return sparse.csc_matrix(exp_dense)
        # Large matrices: use sparse implementation
        return sparse.linalg.expm(A)