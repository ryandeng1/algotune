from typing import Any
import faiss
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        vectors = np.asarray(problem['vectors'], dtype=np.float32)
        k = problem['k']
        dim = vectors.shape[1]

        # Train k‑means (Faiss is already highly optimised)
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        # Assign each vector to the nearest centroid
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)

        # Compute mean squared error
        quantization_error = float(np.mean(distances))

        return {
            'centroids': centroids.tolist(),
            'assignments': assignments.flatten().tolist(),
            'quantization_error': quantization_error
        }