from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Project a symmetric matrix onto the cone of positive semidefinite matrices.

        Parameters
        ----------
        problem : dict
            Dictionary containing the symmetric matrix A under key "A".

        Returns
        -------
        dict
            Dictionary with key "X" containing the projected matrix.
        """
        # Ensure we are working with a numpy array
        A = np.asarray(problem["A"])

        # For symmetric matrices, eigh is more efficient than eig
        eigvals, eigvecs = np.linalg.eigh(A)

        # Zero out negative eigenvalues
        eigvals = np.maximum(eigvals, 0)

        # Reconstruct the projected matrix
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return {"X": X}
