# solver.py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    """
    Optimized solver for computing the exponential of a sparse matrix.
    Uses SciPy's efficient scaling‑and‑squaring algorithm for sparse matrices.
    """

    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        """
        Compute exp(A) for the sparse matrix A provided in the problem dict.

        Parameters
        ----------
        problem : dict[str, sparse.spmatrix]
            Dictionary containing the key 'matrix' mapped to the sparse matrix A.

        Returns
        -------
        sparse.spmatrix
            The matrix exponential exp(A) as a sparse matrix.
        """
        A = problem["matrix"]

        # Ensure we work with a CSC matrix for better performance
        if not sparse.isspmatrix_csc(A):
            A = A.tocsc()

        # Compute the matrix exponential efficiently
        return expm(A)