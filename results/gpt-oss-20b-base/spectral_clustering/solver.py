import numpy as np
from numpy.linalg import eigh
from sklearn.cluster import KMeans

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Fast spectral clustering using NumPy and scikit‑learn's KMeans.

        Parameters
        ----------
        problem : dict
            Must contain:
                - 'similarity_matrix' : square numpy array
                - 'n_clusters'       : positive integer

        Returns
        -------
        dict
            {'labels': labels_array}
        """
        W = problem["similarity_matrix"]
        k = problem["n_clusters"]

        n = W.shape[0]
        if n == 0:
            return {"labels": np.empty((0,), dtype=int)}

        # Sanity checks (minimal)
        if W.shape[0] != W.shape[1] or k < 1:
            raise ValueError("Invalid input")

        if k >= n:
            return {"labels": np.arange(n)}

        # Construct unnormalized Laplacian L = D - W
        D = np.sum(W, axis=1)
        L = np.diag(D) - W

        # Compute first k eigenvectors of L
        # eigh returns ascending order
        _, eigvecs = eigh(L, subset_by_index=[0, k-1])
        # normalize rows to unit length (for consistency)
        norm = np.linalg.norm(eigvecs, axis=1, keepdims=True)
        norm[norm == 0] = 1
        X = eigvecs / norm

        # k-means clustering on the eigenvector embedding
        # Use deterministic seed for reproducibility
        km = KMeans(n_clusters=k, n_init=10, random_state=42, algorithm='elkan')
        labels = km.fit_predict(X)

        return {"labels": labels}