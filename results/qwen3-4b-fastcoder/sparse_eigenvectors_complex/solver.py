from typing import Any
import numpy as np
from scipy import sparse


class Solver:
    def solve(self, problem: dict[str, Any]) -> list[complex]:
        A = problem["matrix"]
        k = problem["k"]
        N = A.shape[0]
        v0 = np.ones(N, dtype=A.dtype)

        eigenvalues, eigenvectors = sparse.linalg.eigs(
            A,
            k=k,
            v0=v0,
            maxiter=N * 200,
            ncv=max(2 * k + 1, 20),
        )

        return eigenvectors.T.tolist()