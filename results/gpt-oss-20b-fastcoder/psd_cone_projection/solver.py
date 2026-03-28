import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Project a symmetric matrix onto the Positive Semidefinite cone.

        Parameters
        ----------
        problem : dict
            Must contain key 'A' : symmetric matrix (n×n).

        Returns
        -------
        dict
            Contains key 'X' : the PSD projection of 'A'.
        """
        # View the matrix as a float64 NumPy array (no copy if already correct)
        A = np.asarray(problem["A"], dtype=np.float64, order="C")

        # Symmetry is assumed; use eigh for efficiency on symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(A)

        # Threshold eigenvalues to non‑negative
        eigvals.clip(min=0, out=eigvals)

        # Reconstruct the PSD matrix: X = V * diag(eigvals) * V^T
        # Use broadcasting to avoid forming an explicit diagonal matrix
        X = (eigvecs * eigvals) @ eigvecs.T
        return {"X": X}