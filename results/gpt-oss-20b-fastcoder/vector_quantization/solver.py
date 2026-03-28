import faiss
import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Perform vector quantization with Faiss k-means.
        """
        # Convert to contiguous float32 array
        vectors = np.asarray(problem["vectors"], dtype=np.float32, order="C")
        k = problem["k"]
        dim = vectors.shape[1]

        # Train k-means with fewer iterations if possible
        kmeans = faiss.Kmeans(dim, k, niter=50, verbose=False, seed=123)
        kmeans.train(vectors)

        # Retrieve centroids
        centroids = kmeans.centroids

        # Build index and search for nearest centroids
        idx = faiss.IndexFlatL2(dim)
        idx.add(centroids)
        distances, assignments = idx.search(vectors, 1)

        # Return results
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(np.mean(distances)),
        }