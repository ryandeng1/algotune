import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Projects a symmetric matrix A onto the cone of positive semi‑definite matrices.
        """
        # Ensure we have a NumPy array (copy if needed)
        A = np.asarray(problem["A"])
        # Use the efficient Hermitian eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        # Set negative eigenvalues to zero (projection onto PSD)
        eigvals = np.where(eigvals > 0, eigvals, 0.0)
        # Reconstruct the projected matrix: V * diag(eigvals) * V^T
        # Avoid constructing an explicit diagonal matrix
        X = (eigvecs * eigvals) @ eigvecs.T
        return {"X": X}