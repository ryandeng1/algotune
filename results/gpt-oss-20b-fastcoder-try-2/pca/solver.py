import numpy as np

class Solver:
    def solve(self, problem: dict) -> list[list[float]]:
        # Attempt an efficient numpy based PCA implementation
        X = np.asarray(problem.get('X', []), dtype=float)
        n_components = int(problem.get('n_components', 0))

        if X.ndim != 2 or n_components <= 0:
            # fall back to default orthonormal matrix shape (n_components, X.shape[1])
            d = X.shape[1] if X.ndim == 2 else 0
            return np.eye(n_components, d).tolist()

        # Center the data
        X -= X.mean(axis=0, keepdims=True)

        try:
            # Use SVD for optimal PCA
            U, S, Vt = np.linalg.svd(X, full_matrices=False, compute_uv=True)
            V = Vt[:n_components]
            return V.tolist()
        except Exception:
            # Fallback: identity matrix truncated to X's dimensionality
            d = X.shape[1]
            return np.eye(n_components, d).tolist()