from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        A = problem['matrix']
        k = problem['k']
        N = A.shape[0]

        # Ensure k does not exceed permissible dimension
        if k > N - 1:
            k = N - 1

        v0 = np.ones(N, dtype=A.dtype)  # initial guess for eigs
        eigvals, eigvecs = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        # Order by decreasing modulus of eigenvalues
        idx = np.argsort(-np.abs(eigvals))
        return [eigvecs[:, i] for i in idx]