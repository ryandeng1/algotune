import numpy as np
import faiss
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vector quantization with Faiss KMeans.

        Parameters
        ----------
        problem : dict
            Must contain:
            - 'vectors': List[List[float]] or np.ndarray of shape (n, d)
            - 'k'       : int, number of centroids

        Returns
        -------
        dict
            {
                'centroids'        : list of centroids (k × d),
                'assignments'      : list of assignments (len n),
                'quantization_error' : float, mean squared error
            }
        """
        # Convert to float32 numpy array once
        vectors = np.asarray(problem['vectors'], dtype=np.float32)
        k = int(problem['k'])
        n, dim = vectors.shape

        # Use the fastest possible KMeans settings
        kmeans = faiss.Kmeans(
            dim,
            k,
            niter=20,          # fewer iterations, good enough for most cases
            nredo=1,           # no redundant restarts
            verbose=False,
            seed=0
        )
        kmeans.train(vectors)

        centroids = kmeans.centroids
        # Build flat index for nearest-centroid search
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)

        # Query nearest centroid for all vectors
        distances, assignments = index.search(vectors, 1)

        mean_error = float(np.mean(distances))

        return {
            'centroids': centroids.tolist(),
            'assignments': assignments.reshape(-1).tolist(),
            'quantization_error': mean_error
        }