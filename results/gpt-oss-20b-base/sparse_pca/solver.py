import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fast greedy sparse PCA by truncating the leading eigenvectors.
        """
        # Parse input
        A = np.asarray(problem["covariance"], dtype=float)
        n_components = int(problem["n_components"])
        sparsity = float(problem["sparsity_param"])

        # Eigen-decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(A)
        # Keep only positive eigenvalues (noise filtering)
        pos = eigvals > 0
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]
        # Sort by decreasing magnitude
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Select the required number of components
        n_used = min(len(eigvals), n_components)
        components = eigvecs[:, :n_used]

        # Enforce sparsity by zeroing out small coefficients
        # sparsity is interpreted as the proportion of non‑zero entries
        n_nonzero = max(1, int(np.round(sparsity * A.shape[0])))
        for j in range(components.shape[1]):
            col = components[:, j]
            # Find indices of the largest |coefficients|
            idx = np.argsort(np.abs(col))[-n_nonzero:]
            mask = np.zeros_like(col, dtype=bool)
            mask[idx] = True
            components[:, j] = col * mask

        # Ensure each component has unit norm (optional)
        norms = np.linalg.norm(components, axis=0)
        if np.any(norms > 0):
            components = components / norms

        # Compute explained variance for each component
        explained_variance = []
        for j in range(components.shape[1]):
            c = components[:, j]
            var = float(c.T @ A @ c)
            explained_variance.append(var)

        return {
            "components": components.tolist(),
            "explained_variance": explained_variance,
        }