import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]
        n_samples = problem["n_samples"]
        
        # Check for trivial cases
        if n_clusters >= n_samples:
            labels = np.arange(n_samples)
            return {"labels": labels.tolist(), "n_clusters": n_clusters}
        
        # Validate similarity matrix
        if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("Invalid similarity matrix provided.")
        
        # Compute degree matrix (D)
        deg = np.sum(similarity_matrix, axis=1)
        
        # Compute D^{-1/2} with small epsilon for division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            deg_sqrt = np.sqrt(np.maximum(deg, 1e-12))
            D_inv_sqrt = np.diag(1.0 / deg_sqrt)
        
        # Compute normalized Laplacian L = I - D^{-1/2} S D^{-1/2}
        L = np.eye(n_samples) - D_inv_sqrt @ similarity_matrix @ D_inv_sqrt
        
        # Compute eigenvalues and eigenvectors
        evals, evecs = eigh(L)
        
        # Take the first n_clusters eigenvectors (excluding the zero eigenvalue)
        eigenvectors = evecs[:, 1:1+n_clusters]
        
        # Project the data onto these eigenvectors
        projected_data = evecs[:n_samples, 1:1+n_clusters]
        
        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(projected_data)
        
        return {"labels": labels.tolist(), "n_clusters": n_clusters}
