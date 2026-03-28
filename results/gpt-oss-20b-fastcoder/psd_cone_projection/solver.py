import numpy as np
from typing import Any, Dict

class Solver:
    """Fast projection of a symmetric matrix onto the PSD cone."""
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        # Use eigh (optimal for symmetric matrices) and avoid unnecessary copies
        A = problem["A"]
        eigvals, eigvecs = np.linalg.eigh(A)

        # Zero out negative eigenvalues in-place for speed
        eigvals[eigvals < 0] = 0.0

        # Scale eigenvectors by the eigenvalues
        # This uses broadcasting and eliminates an explicit diag matrix
        proj = eigvecs * eigvals
        X = proj @ eigvecs.T

        return {"X": X}