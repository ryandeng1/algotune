import faiss
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Vector quantisation using Faiss K‑means."""
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        dim = vectors.shape[1]

        # Perform K‑means clustering
        kmeans = faiss.Kmeans(dim, k, niter=200, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        # Assign each vector to its nearest centroid
        quantizer = faiss.IndexFlatL2(dim)
        quantizer.add(centroids)
        distances, assignments = quantizer.search(vectors, 1)

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(np.mean(distances)),
        }