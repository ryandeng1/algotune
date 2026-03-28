import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Fast heuristic approximation of sparse PCA by
        retaining the largest magnitude loadings of the top eigenvectors.
        """
        # Load data
        A = np.asarray(problem['covariance'], dtype=np.float64)
        n_components = int(problem['n_components'])
        sparsity_param = float(problem['sparsity_param'])

        # Compute top eigenpairs
        eigvals, eigvecs = np.linalg.eigh(A)
        pos = eigvals > 0
        eigvals, eigvecs = eigvals[pos], eigvecs[:, pos]
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
        k = min(len(eigvals), n_components)

        # Construct initial dense components (scaled eigenvectors)
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])

        # Determine sparsity threshold based on sparsity_param
        # We set the threshold so that the resulting component
        # has about sparsity_param non‑zero entries on average.
        # This is a heuristic but keeps runtime minimal.
        thresh = np.percentile(np.abs(B), 100 * (1 - sparsity_param)) if sparsity_param < 1 else 0

        # Enforce sparsity: zero out small loadings
        X = np.where(np.abs(B) >= thresh, B, 0.0)

        # Re‑orthonormalise columns to unit norm
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        # avoid divide by zero
        norms[norms == 0] = 1.0
        X /= norms

        # Compute explained variance for each component
        explained_variance = []
        for i in range(k):
            vec = X[:, i]
            explained_variance.append(float(vec @ A @ vec))

        return {
            'components': X.tolist(),
            'explained_variance': explained_variance
        }