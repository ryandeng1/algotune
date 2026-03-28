import numpy as np
import faiss
from typing import Any

class Solver:
    """
    Vector Quantisation using Faiss k‑means.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract and cast efficiently
        vectors = np.asarray(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        dim = vectors.shape[1]

        # K‑means: fewer iterations give a good trade‑off in most tests.
        kmeans = faiss.Kmeans(dim, k, niter=20, verbose=False, seed=0)
        kmeans.train(vectors)

        # Direct distance/assignment via Faiss flat index
        centroids = kmeans.centroids
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)

        # Mean squared error (already squared distance)
        mse = float(np.mean(distances))

        # Convert to pure Python objects for the expected API
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": mse,
        }