import numpy as np
import faiss

class Solver:
    def solve(self, problem):
        vectors = np.asarray(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)

        centroids = kmeans.centroids
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)

        distances, assignments = index.search(vectors, 1)
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(np.mean(distances))
        }