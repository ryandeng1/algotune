import numpy as np
import faiss

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        """
        Perform vector quantization using Faiss's K-means implementation.
        The algorithm minimizes the mean squared Euclidean distance between
        each input vector and the nearest centroid.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "vectors": list of vectors (list[list[float]])
                - "k": desired number of centroids
        **kwargs : Any
            Additional arguments are ignored (kept for API compatibility).

        Returns
        -------
        dict
            Dictionary containing:
                - "centroids": list of k centroids
                - "assignments": list of cluster indices for each vector
                - "quantization_error": mean squared error
        """
        vectors = np.asarray(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        dim = vectors.shape[1]

        # K-means clustering with Faiss
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)

        centroids = kmeans.centroids

        # Assign each vector to the nearest centroid
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)

        quantization_error = float(np.mean(distances))

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": quantization_error,
        }
