# solver.py
import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> np.ndarray:
        """
        Perform Principal Component Analysis (PCA) using the SVD of the centered data.
        Returns an array V of shape (n_components, n_features) where each row is a
        principal component vector with unit norm and orthogonal to the others.
        """
        X = np.asarray(problem["X"], dtype=np.float64)
        n_components = int(problem["n_components"])

        # Center the data
        X -= X.mean(axis=0, keepdims=True)

        # SVD: X = U * S * Vh
        # The rows of Vh (or columns of V) are the principal components.
        # We want the first n_components rows.
        # Use full_matrices=False for efficiency.
        _, _, Vh = np.linalg.svd(X, full_matrices=False)

        # Take first n_components rows
        V = Vh[:n_components, :]

        # Normalize rows (they should already be unit norm, but ensure numeric stability)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / norms

        return V
