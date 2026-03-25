import numpy as np

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Fast approximate solution for sparse PCA.
        Computes leading eigenvectors, then sparsifies them by thresholding
        small loadings. The columns are renormalised to satisfy the unit norm
        constraint.  The explained variance is computed from the covariance
        matrix.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "covariance" : 2-D list or array representing the covariance matrix
                - "n_components" : int, number of components to extract
                - "sparsity_param" : float, threshold factor for sparsity

        Returns
        -------
        dict
            Dictionary with keys:
                - "components" : 2-D list of shape (n_features, n_components)
                - "explained_variance" : list of float values
        """
        # Load problem data
        A = np.array(problem["covariance"], dtype=float)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])

        n = A.shape[0]

        # ---- Eigen-decomposition ----
        eigvals, eigvecs = np.linalg.eigh(A)
        pos = eigvals > 0
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]
        # sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        k = min(len(eigvals), n_components)
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # 2-D array for output (n x n_components)
        X = np.zeros((n, n_components))

        # ---- Sparsification & normalisation ----
        for j in range(k):
            col = B[:, j].copy()
            # threshold: retain elements larger than sparsity_param * max_abs
            max_abs = np.max(np.abs(col))
            thresh = sparsity_param * max_abs
            col[np.abs(col) < thresh] = 0.0
            # renormalise to unit norm if not zero
            norm = np.linalg.norm(col)
            if norm > 0:
                col /= norm
            X[:, j] = col

        # If requested more components than available eigenvectors
        if n_components > k:
            X[:, k:] = np.zeros((n, n_components - k))

        # ---- Explained variance ----
        explained_variance = []
        for j in range(n_components):
            v = X[:, j]
            var = float(v.T @ A @ v)
            explained_variance.append(var)

        return {
            "components": X.tolist(),
            "explained_variance": explained_variance,
        }
