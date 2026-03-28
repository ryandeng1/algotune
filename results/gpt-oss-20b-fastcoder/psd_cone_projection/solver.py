from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Project a symmetric matrix onto the positive semidefinite cone by
        clamping negative eigenvalues to zero.
        """
        A = np.asarray(problem["A"], dtype=float)
        # Since A is symmetric, use eigh for better performance and stability.
        eigvals, eigvecs = np.linalg.eigh(A)
        # Clamp negative eigenvalues to zero.
        eigvals = np.clip(eigvals, 0, None)
        # Reconstruct the projected matrix without forming a diagonal matrix.
        # (eigvecs @ (eigvecs.T * eigvals)) is equivalent to V @ diag(eigvals) @ V.T
        X = eigvecs @ (eigvecs.T * eigvals)
        return {"X": X}