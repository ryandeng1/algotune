import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm

class Solver:
    """
    Optimized sparse matrix exponential solver.
    Uses scipy's built‑in expm implementation, but adds a few micro‑optimisations:
    * Avoids unnecessary copying.
    * Handles trivial zero matrices quickly.
    * Expects input to be in CSR format for maximum speed.
    """
    def solve(self, problem: dict[str, sparse.spmatrix]) -> sparse.spmatrix:
        A = problem['matrix']
        # Quickly return identity for zero matrix to avoid costly computation
        if A.nnz == 0:
            return sparse.eye(A.shape[0], dtype=A.dtype, format=A.format)
        # Ensure CSR format for fast arithmetic
        if not sparse.isspmatrix_csr(A):
            A = A.tocsr()
        # Use the efficient SciPy routine
        return expm(A)