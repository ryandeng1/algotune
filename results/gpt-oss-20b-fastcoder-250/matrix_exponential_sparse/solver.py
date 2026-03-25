import numpy as np
from scipy import sparse
from scipy.linalg import expm


class Solver:
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Compute the matrix exponential of a sparse matrix.
        The input is expected to be a CSC sparse matrix. For performance
        reasons the matrix is converted to dense, the dense exponential
        is computed with scipy.linalg.expm and then converted back to
        a sparse CSC matrix.
        """
        A = problem["matrix"]
        # Convert to dense NumPy array
        dense = A.toarray()
        # Compute dense matrix exponential
        exp_dense = expm(dense)
        # Convert back to sparse CSC matrix
        return sparse.csc_matrix(exp_dense)
