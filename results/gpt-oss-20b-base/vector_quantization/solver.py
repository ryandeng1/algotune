import faiss
import numpy as np
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the Vector Quantization problem using the Faiss library.

        Parameters
        ----------
        problem:
            Dictionary containing:
            - "vectors": Iterable of vectors (each a list or array of length ``dim``)
            - "k": Number of centroids to find

        Returns
        -------
        Dictionary with keys:
            - "centroids": 2‑D list of the computed centroids (k × dim)
            - "assignments": 1‑D list mapping each input vector to its nearest centroid
            - "quantization_error": Mean squared error across all points
        """
        # Convert input to a contiguous float32 numpy array once
        vectors = np.asarray(problem["vectors"], dtype=np.float32, order="C")
        k = int(problem["k"])
        dim = vectors.shape[1]

        # Train k‑means with a deterministic seed for reproducibility
        kmeans = faiss.Kmeans(d=dim, k=k, niter=200, seed=12345, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        # Build a flat index for exact L2 nearest‑centroid search
        flat_index = faiss.IndexFlatL2(dim)
        flat_index.add(centroids)

        # Search each vector for its closest centroid
        distances, assignments = flat_index.search(vectors, 1)

        # Quantization error is the mean squared L2 distance
        quantization_error = float(np.mean(distances))

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": quantization_error,
        }