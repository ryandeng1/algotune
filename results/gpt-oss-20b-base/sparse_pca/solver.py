import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Fast approximate solution for sparse PCA.

        The method follows these steps:
            1. Compute the eigen-decomposition of the covariance matrix A.
            2. Construct the dense principal components B from the top k eigenvectors.
            3. Apply element‑wise hard thresholding to introduce sparsity.
            4. Renormalise each column to satisfy the unit‑norm constraint.
            5. Compute explained variance for each component.
        """
        # Extract data
        A = np.array(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        n = A.shape[0]

        # eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        pos = eigvals > 1e-12
        eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]
        # descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        k = min(n_components, len(eigvals))
        # Dense components
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # Hard thresholding: keep entries larger than sparsity_param
        threshold = sparsity_param
        X = np.where(np.abs(B) >= threshold, B, 0.0)

        # Renormalise columns to unit norm (if zero column, keep zero)
        norms = np.linalg.norm(X, axis=0)
        nonzero = norms > 1e-12
        X[:, nonzero] /= norms[nonzero]

        # Compute explained variance
        explained = [(X[:, i].T @ A @ X[:, i]).item() for i in range(n_components)]

        return {"components": X.tolist(), "explained_variance": explained}
