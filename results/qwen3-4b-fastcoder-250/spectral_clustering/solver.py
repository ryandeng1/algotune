import numpy as np
from scipy.cluster.vq import kmeans

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]
        n_samples = problem["n_samples"]
        
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Invalid similarity matrix provided.")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("Invalid number of clusters provided.")
        
        if n_clusters >= n_samples:
            return {"labels": list(range(n_samples)), "n_clusters": n_clusters}
        if n_clusters == 1:
            return {"labels": [0] * n_samples, "n_clusters": n_clusters}
        
        S = np.clip(similarity_matrix, 0.0, 1.0).astype(float)
        np.fill_diagonal(S, 0.0)
        
        deg = np.sum(S, axis=1)
        deg = np.maximum(deg, 1e-12)
        dinv_sqrt = 1.0 / np.sqrt(deg)
        
        Dhalf = np.diag(dinv_sqrt)
        L = np.eye(n_samples) - Dhalf @ S @ Dhalf
        
        evals, evecs = np.linalg.eigh(L)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        U = evecs[:, 1:1+n_clusters]
        
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        U_norm = U / norms
        
        centroids, labels = kmeans(U_norm, n_clusters)
        
        return {"labels": labels.tolist(), "n_clusters": n_clusters}
