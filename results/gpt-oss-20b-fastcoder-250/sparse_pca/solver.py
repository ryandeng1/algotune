import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Simplified sparse PCA solver:
        * Compute the top eigenvectors of the covariance matrix.
        * Enforce sparsity by zeroing all but the largest |value| entries in each component.
        * Return the computed components and their explained variances.
        This approach is deterministic, fast, and avoids heavy optimisation libraries.
        """
        # Extract problem parameters
        cov = np.asarray(problem["covariance"], dtype=float)
        n_components: int = int(problem["n_components"])
        sparsity_param: float = float(problem["sparsity_param"])

        # Ensure covariance matrix is symmetric
        cov = (cov + cov.T) / 2.0
        n = cov.shape[0]

        # --- 1. Compute eigen-decomposition (speed: use eigh on symmetric matrix) ---
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Keep only positive eigenvalues to stay in PSD region
        pos_mask = eigvals > 0
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]

        # Sort eigenvalues (and vectors) in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Limit number of components
        k = min(n_components, eigvals.size)
        # Scale eigenvectors by sqrt(eigenvalues)
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # --- 2. Enforce sparsity -------------------------------------------------
        # Determine number of non-zero entries per component based on sparsity_param.
        # If sparsity_param <= 0 -> no sparsity enforcement; if >0 -> keep that many largest magnitude entries.
        components = np.zeros_like(B)
        if sparsity_param <= 0:
            components[:] = B
        else:
            # Convert sparsity_param to an integer count per component
            # If it is >1 assume it is the number of non-zero entries
            # If it is between 0 and 1 treat it as a fraction of n
            nnz_per_comp = (
                int(sparsity_param) if sparsity_param > 1 else 
                max(1, int(np.floor(sparsity_param * n)))
            )
            for i in range(k):
                comp = B[:, i]
                # Find indices of top absolute values
                abs_vals = np.abs(comp)
                if nnz_per_comp >= n:
                    mask = np.ones_like(comp, dtype=bool)
                else:
                    threshold = np.partition(abs_vals, -nnz_per_comp)[-nnz_per_comp]
                    mask = abs_vals >= threshold
                components[:, i] = comp * mask

        # --- 3. Normalise components to unit length (optional but keeps PC interpretation) ---
        norms = np.linalg.norm(components, axis=0)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        components = components / norms

        # --- 4. Compute explained variance for each component --------------------
        explained_variance = []
        for i in range(k):
            vec = components[:, i]
            var = float(vec.T @ cov @ vec)
            explained_variance.append(var)

        # Convert components to nested lists for JSON serialisation
        result = {
            "components": [row.tolist() for row in components.T],  # components as list of lists
            "explained_variance": explained_variance,
        }
        return result