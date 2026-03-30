import numpy as np
from typing import Any, Dict

class Solver:
    """
    Performance‑optimized solver for PSD cone projection.
    """
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        A = problem["A"]
        # Ensure we work on a float array (input may be immutable)
        if not isinstance(A, np.ndarray):
            A = np.asarray(A, dtype=np.float64)
        # Eigen‑decomposition using the symmetric routine (faster than eig)
        eigvals, eigvecs = np.linalg.eigh(A)
        # Clamp negative eigenvalues to zero
        eigvals = np.maximum(eigvals, 0.0)
        # Reconstruct X without building a large diagonal matrix
        # (A = V * diag(eigvals) * V.T) -> (V * eigvals) @ V.T
        X = (eigvecs * eigvals) @ eigvecs.T
        return {"X": X}