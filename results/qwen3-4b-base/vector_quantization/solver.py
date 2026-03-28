from typing import Any
import faiss
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        vectors = problem["vectors"]
        k = problem["k"]
        vectors = np.array(vectors).astype(np.float32)
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        distances, assignments = kmeans.assign(vectors)

        quantization_error = np.mean(distances)
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.tolist(),
            "quantization_error": quantization_error,
        }