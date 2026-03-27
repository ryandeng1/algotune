import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Projects a symmetric matrix onto the positive semidefinite cone.

        Parameters
        ----------
        problem : dict
            Must contain key 'A' with a symmetric NumPy array.

        Returns
        -------
        dict
            A dictionary with key 'X', the projected matrix.
        """
        # Ensure a read‑only view and avoid unnecessary copies
        A = np.asarray(problem["A"], dtype=np.float64, order="C")

        # Eigen‑decomposition for symmetric matrices (fast path)
        w, v = np.linalg.eigh(A, UPLO="U")       # w: array of eigenvalues, v: eigenvectors
        w = np.maximum(w, 0.0)                    # clip to non‑negative

        # Reconstruct the projected matrix without creating an intermediate diagonal matrix
        X = (v * w) @ v.T                         # equivalent to v @ diag(w) @ v.T

        return {"X": X}