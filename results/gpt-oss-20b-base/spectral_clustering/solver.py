import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh

class Solver:
    def solve(self, problem, **kwargs):
        """Fast spectral clustering using normalized Laplacian eigendecomposition
        followed by k‑means.  Handles edge cases from the validator."""

        S = np.asarray(problem["similarity_matrix"], dtype=float)
        n_clusters = int(problem["n_clusters"])
        n = S.shape[0]

        # Edge cases
        if n == 0:
            return {"labels": [], "n_clusters": n_clusters}
        if n_clusters == 1:
            return {"labels": [0] * n, "n_clusters": n_clusters}
        if n_clusters >= n:
            return {"labels": list(range(n)), "n_clusters": n_clusters}

        # Ensure symmetry and zero diagonal
        S = (S + S.T) / 2.0
        np.fill_diagonal(S, 0.0)

        # Degrees and normalized Laplacian
        deg = S.sum(axis=1)
        with np.errstate(divide="ignore"):
            inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D_half = np.diag(inv_sqrt)
        L = np.eye(n) - D_half @ S @ D_half

        # Compute first k eigenvectors of Laplacian
        # Use sparse eigsh for speed; fallback to dense eigh if fails
        try:
            evals, evecs = eigsh(L, k=n_clusters, which='SM', return_eigenvectors=True)
            U = evecs
        except Exception:
            evals, evecs = np.linalg.eigh(L)
            evecs_sorted = evecs[:, np.argsort(evals)]
            U = evecs_sorted[:, :n_clusters]

        # Normalize rows of U
        U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
        U_norm[np.isnan(U_norm)] = 0.0

        # k‑means on the rows of U_norm
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42, max_iter=300)
        labels = km.fit_predict(U_norm)

        return {"labels": labels.tolist(), "n_clusters": n_clusters}
