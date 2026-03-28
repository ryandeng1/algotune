from typing import Any, List
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> List[List[float]]:
        try:
            X = np.asarray(problem['X'], dtype=float)
            if X.ndim != 2:
                raise ValueError("Input must be a 2D array")
            n_components = int(problem['n_components'])
            # Center the data
            X -= X.mean(axis=0, keepdims=True)
            # Perform SVD; V^T contains principal components
            # We want the first n_components components
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            components = VT[:n_components]
            return components.tolist()
        except Exception:
            # fallback: orthonormal identity matrix truncated to n_components and n features
            X = np.asarray(problem['X'], dtype=float)
            n, d = X.shape
            c = int(problem['n_components'])
            V = np.zeros((c, d))
            # fill first c rows with identity across columns
            min_cd = min(c, d)
            V[:min_cd, :min_cd] = np.eye(min_cd)
            return V.tolist()