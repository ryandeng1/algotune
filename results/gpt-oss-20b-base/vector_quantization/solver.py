from typing import Any
import numpy as np
import faiss

class Solver:
    # Faiss init; use GPU if available (fallback to CPU)
    def __init__(self):
        self.use_gpu = faiss.get_num_gpus() > 0
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.kmeans_builder = faiss.GpuIndexFlatL2(res, self._dim)  # placeholder, will be set in solve

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        vectors = np.asarray(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=50, verbose=False, nredo=1)
        kmeans.train(vectors)

        centroids = kmeans.centroids
        # Build index with centroids
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)

        distances, assignments = index.search(vectors, 1)
        quantization_error = float(np.mean(distances))

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": quantization_error,
        }