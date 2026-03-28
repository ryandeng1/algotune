import numpy as np

class Solver:
    """
    A fast, approximate sparse PCA solver.
    Computes the ordinary PCA, then keeps only the largest
    `sparsity_param` (interpreted as a proportion of non‑zeros)
    elements in each component and renormalises.
    """

    def solve(self, problem: dict) -> dict:
        # Extract data
        cov = np.asarray(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        # Compute eigen decomposition (only the leading part is needed)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Keep only positive eigenvalues
        pos = eigvals > 0
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]

        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Take the first n_components principal directions
        k = min(len(eigvals), n_components)
        comps = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # Sparsify each component: keep only the largest `sparsity_param` proportion
        if sparsity_param < 1:
            keep = max(1, int(np.ceil(k * sparsity_param)))
            # For each component, zero out all but the `keep` largest absolute entries
            for j in range(k):
                col = comps[:, j]
                if np.count_nonzero(col) > keep:
                    # mask for the largest absolute values
                    mask = np.abs(col) >= np.partition(np.abs(col), -keep)[-keep]
                    col *= mask
                    # renormalise
                    norm = np.linalg.norm(col)
                    if norm > 0:
                        col /= norm
                comps[:, j] = col

        # Compute explained variance for each component
        explained = []
        for j in range(k):
            vec = comps[:, j]
            variance = float(vec.T @ cov @ vec)
            explained.append(variance)

        return {"components": comps.tolist(), "explained_variance": explained}