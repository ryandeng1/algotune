import numpy as np
from typing import Any, Dict
from sklearn.cluster import KMeans
from scipy.linalg import eigh

class Solver:
    """
    An efficient spectral clustering implementation that avoids the full
    sklearn SpectralClustering machinery.

    The strategy is:
        1. Compute the unnormalized graph Laplacian L = D - W.
        2. Extract the first n_clusters eigenvectors of L (corresponding to the
           smallest eigenvalues).
        3. Run k-means on the rows of the eigenvector matrix to obtain cluster
           labels.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        W = problem["similarity_matrix"]
        k = problem["n_clusters"]

        # --- Basic validation -------------------------------------------------
        if not isinstance(W, np.ndarray) or W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError("similarity_matrix must be a square numpy array.")
        if not isinstance(k, int) or k < 1:
            raise ValueError("n_clusters must be a positive integer.")
        N = W.shape[0]
        if N == 0:
            return {"labels": np.array([], dtype=int)}

        # --- Special trivial cases --------------------------------------------
        if k >= N:
            return {"labels": np.arange(N, dtype=int)}

        # --- Compute degree matrix and unnormalized Laplacian ---------------
        d = np.sum(W, axis=1)
        L = np.diag(d) - W  # N x N dense matrix

        # --- Compute first k eigenvectors (smallest eigenvalues)-----------
        # eigh returns eigenvalues in ascending order
        vals, vecs = eigh(L, subset_by_index=[0, k - 1])
        X = vecs  # rows are the eigenvectors for each node

        # --- k-means on the rows of X --------------------------------------
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300, tol=1e-4)
        labels = kmeans.fit_predict(X)

        return {"labels": labels}