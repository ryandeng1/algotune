# solver.py
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

class Solver:
    """Fast sparse eigenvalue solver."""

    def solve(self, problem: dict[str, Any]) -> list[complex]:
        """
        Compute the `k` eigenvectors of the largest-magnitude eigenvalues of a square
        sparse matrix `A` given in COO, CSR, CSC or any scipy sparse format.

        Parameters
        ----------
        problem : dict
            Must contain:
            - ``matrix``: A scipy sparse matrix.
            - ``k``: Number of dominant eigenvectors to return (int).

        Returns
        -------
        list[complex]
            Eigenvectors sorted by decreasing |λ|.  Each entry is a 1‑D :class:`numpy.ndarray`
            of dtype matching `A.dtype`.  The list length is ``min(k, N)`` where N is the
            dimension of the matrix.
        """
        A: sparse.spmatrix = problem["matrix"]
        k: int = problem["k"]
        N: int = A.shape[0]

        # Sanity check: ensure we don't request more eigenpairs than allowed
        if k <= 0:
            raise ValueError("k must be positive")
        if k >= N:
            # The solver can only return at most N-1 eigenvalues, so
            # reduce the request to the max admissible value.
            k = N - 1

        # Provide a useful initial guess
        v0 = np.full(N, 1, dtype=A.dtype)

        # Run the ARPACK routine via scipy sparse.linalg
        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        # Sort by decreasing eigenvalue magnitude
        idx = np.argsort(-np.abs(eigenvalues))
        eigenvectors = eigenvectors[:, idx]

        # Return a list of 1‑D arrays (vectors)
        return [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]