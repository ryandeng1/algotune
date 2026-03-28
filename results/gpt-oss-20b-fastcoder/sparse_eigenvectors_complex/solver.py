from typing import Any
import numpy as np
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        A = problem['matrix']
        k = problem['k']
        N = A.shape[0]

        # initial guess: ones vector of the correct dtype
        v0 = np.ones(N, dtype=A.dtype)

        # eigs with reasonably tight default parameters
        eigvals, eigvecs = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20)
        )

        # sort indices by descending absolute value of eigenvalues
        idx = np.argsort(-np.abs(eigvals))

        # build result list in sorted order
        return [eigvecs[:, i] for i in idx]