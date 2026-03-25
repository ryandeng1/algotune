import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Vector Quantization problem using Faiss's KMeans implementation.
        The method finds `k` centroids that minimize the mean squared error between each input vector
        and its assigned centroid. It then assigns each vector to the nearest centroid.
        The returned quantization error is the average squared Euclidean distance over all vectors.
        """
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        dim = vectors.shape[1]

        # Perform KMeans clustering with Faiss
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False, seed=0)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        # Build a flat index for nearest-centroid search
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)

        quantization_error = float(np.mean(distances))

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": quantization_error,
        }
