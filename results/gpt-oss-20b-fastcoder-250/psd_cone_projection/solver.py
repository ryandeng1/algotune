import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Projects a symmetric matrix A onto the cone of symmetric positive semidefinite matrices.

        Parameters
        ----------
        problem : dict
            Dictionary containing key 'A' with a symmetric matrix.

        Returns
        -------
        dict
            Dictionary with key 'X' holding the projected matrix.
        """
        # Convert input to a NumPy array and ensure float dtype
        A = np.asarray(problem["A"], dtype=np.float64)

        # Eigen‑decomposition optimized for symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(A)

        # Set negative eigenvalues to zero (projection onto PSD cone)
        eigvals[eigvals < 0] = 0.0

        # Reconstruct the matrix
        X = (eigvecs * eigvals) @ eigvecs.T

        return {"X": X}
