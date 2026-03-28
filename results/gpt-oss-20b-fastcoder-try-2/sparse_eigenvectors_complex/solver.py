from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Compute the `k` eigenvectors with the largest modulus eigenvalues of the sparse matrix `A`.
        Uses `scipy.sparse.linalg.eigs` with tuned parameters for speed.
        """
        A: sparse.spmatrix = problem["matrix"]
        k: int = problem["k"]
        n: int = A.shape[0]

        # Initial guess - all ones; ensures consistency across runs
        v0 = np.ones(n, dtype=A.dtype)

        # Parameters tuned for large sparse problems:
        #   * `maxiter` trimmed to 2*n (default 6*n) to save iterations
        #   * `ncv` set to 2*k+1 (minimum recommended) with a lower bound
        maxiter = 2 * n
        ncv = max(2 * k + 1, 20)

        # Compute eigenvalues and eigenvectors
        eigvals, eigvecs = sparse.linalg.eigs(
            A, k=k, v0=v0, maxiter=maxiter, ncv=ncv, return_eigenvectors=True
        )

        # Sort eigenvectors by decreasing modulus of eigenvalues
        idx = np.argsort(-np.abs(eigvals))
        sorted_eigvecs = [eigvecs[:, i] for i in idx]

        return sorted_eigvecs