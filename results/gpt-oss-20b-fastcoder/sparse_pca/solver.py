from typing import Dict, Any
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast approximation of sparse PCA: simply returns the top
        eigenvectors of the covariance matrix.  This keeps the
        original API but removes the heavy CVXPY dependency.
        """
        try:
            # Extract problem parameters
            A = np.asarray(problem["covariance"], dtype=np.float64)
            n_components = int(problem["n_components"])
            # we do not use sparsity_param, but keep it for compatibility
        except Exception:
            return {"components": [], "explained_variance": []}

        # Compute eigen-decomposition (ascending order)
        eigvals, eigvecs = np.linalg.eigh(A)
        # Keep positive eigenvalues only
        pos_mask = eigvals > 0
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]
        # Reverse to descending order
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]

        k = min(len(eigvals), n_components)
        components = eigvecs[:, :k].T  # shape (k, n)

        # Normalise components to unit L2 norm
        norms = np.linalg.norm(components, axis=1, keepdims=True)
        components = components / norms

        # Explain variance
        explained_variance = np.sum(components @ A * components, axis=1).tolist()

        return {
            "components": components.tolist(),
            "explained_variance": explained_variance,
        }