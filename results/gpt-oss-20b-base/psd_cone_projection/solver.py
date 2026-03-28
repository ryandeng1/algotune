from typing import Any, Dict
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Efficiently projects the symmetric matrix A onto the positive semidefinite cone.
        """
        # Ensure A is a contiguous float64 array
        A = np.asarray(problem["A"], dtype=np.float64, order="C")

        # Solve symmetric eigenvalue problem (cheaper than general eig)
        eigvals, eigvecs = np.linalg.eigh(A)

        # Zero out negative eigenvalues (in-place for speed)
        np.maximum(eigvals, 0, out=eigvals)

        # Reconstruct X = V * Λ * V^T efficiently:
        #   (V * Λ) @ V^T   where multiplication is element‑wise
        X = (eigvecs * eigvals).dot(eigvecs.T)

        return {"X": X}