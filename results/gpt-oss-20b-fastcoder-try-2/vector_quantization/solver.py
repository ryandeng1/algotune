from typing import Any
import numpy as np
import faiss


class Solver:
    """
    A fast vector‑quantisation solver that uses Faiss' GPU‑friendly K‑means.
    The implementation avoids unnecessary Python overhead and guarantees the same
    output format as the original version.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Quick conversion to a C‑contiguous float32 array
        vectors = np.asarray(problem["vectors"], dtype=np.float32, order="C")
        k = problem["k"]
        dim = vectors.shape[1]

        # Faiss Kmeans – use default options for speed & determinism
        kmeans = faiss.Kmeans(dim, k, niter=100, nredo=1, verbose=False)
        kmeans.train(vectors)

        centroids = kmeans.centroids
        assert centroids.shape[0] == k and centroids.shape[1] == dim

        # Assign each vector to its nearest centroid
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        dists, idx = index.search(vectors, 1)

        # Return results in the requested format
        return {
            "centroids": centroids.tolist(),
            "assignments": idx.flatten().tolist(),
            "quantization_error": float(np.mean(dists)),
        }