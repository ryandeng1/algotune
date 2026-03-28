import numpy as np
from sklearn.cluster import SpectralClustering

class Solver:

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        similarity_matrix = problem['similarity_matrix']
        n_clusters = problem['n_clusters']

        # Simple validation
        if (not isinstance(similarity_matrix, np.ndarray) or
                similarity_matrix.ndim != 2 or
                similarity_matrix.shape[0] != similarity_matrix.shape[1]):
            raise ValueError('Invalid similarity matrix provided.')

        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError('Invalid number of clusters provided.')

        size = similarity_matrix.shape[0]
        if n_clusters >= size:
            labels = np.arange(size)
        elif size == 0:
            labels = np.array([], dtype=int)
        else:
            # Precomputed affinity, use kmeans assignment
            model = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                assign_labels='kmeans',
                random_state=42,
                n_init=10,
            )
            labels = model.fit_predict(similarity_matrix)

        return {'labels': labels}