from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, dict[str, list[list[complex]]]]:
        """
        Solve X @ X = A for the principal matrix square root X.
        Uses eigen‑decomposition to avoid the overhead of scipy.linalg.sqrtm.
        """
        A = problem["matrix"]
        try:
            # Eigen‑decomposition
            w, v = np.linalg.eig(A)
            # Diagonal matrix of square roots of eigenvalues
            sqrt_w = np.sqrt(w)
            # Reconstruct the square root
            Vinv = np.linalg.inv(v)
            X = v @ np.diag(sqrt_w) @ Vinv
        except Exception:
            # Return an empty result on failure
            return {"sqrtm": {"X": []}}
        else:
            # Convert to nested Python lists with complex numbers
            return {"sqrtm": {"X": X.tolist()}}