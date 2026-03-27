import numpy as np
from typing import Any


class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Projects a real symmetric matrix onto the positive semidefinite cone.
        Uses the efficient eigh routine and a single matrix multiplication.

        Args:
            problem: A dictionary with key 'A' containing a symmetric numpy array.

        Returns:
            A dictionary with key 'X' containing the projected matrix.
        """
        A = problem["A"]

        # eigh is faster for symmetric matrices and returns real eigenvalues/vectors
        eigvals, eigvecs = np.linalg.eigh(A)

        # Clip negative eigenvalues to zero
        eigvals = np.maximum(eigvals, 0.0)

        # Scale eigenvector columns by eigenvalues and rebuild X
        # The operation V * diag(l) * V.T can be written as (V * l) @ V.T
        X = (eigvecs * eigvals) @ eigvecs.T

        return {"X": X}