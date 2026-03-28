from typing import Any
import faiss
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        vectors = problem["vectors"]
        k = problem["k"]
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)

        quantization_error = np.mean(distances)
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.ravel().tolist(),
            "quantization_error": quantization_error,
        }