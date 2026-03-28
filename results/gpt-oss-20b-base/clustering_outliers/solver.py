from typing import Any
import numpy as np
import hdbscan

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        # Directly convert the dataset to a NumPy array (no copy if already an array)
        dataset = np.asarray(problem['dataset'])

        # Retrieve clustering parameters with defaults
        min_cluster_size = problem.get('min_cluster_size', 5)
        min_samples = problem.get('min_samples', 3)

        # Fit the HDBSCAN model
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        clusterer.fit(dataset)

        # Extract results
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Prepare the solution dictionary
        num_clusters = int(np.count_nonzero(np.unique(labels[labels != -1])))
        num_noise_points = int(np.count_nonzero(labels == -1))

        return {
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist(),
            'cluster_persistence': persistence.tolist(),
            'num_clusters': num_clusters,
            'num_noise_points': num_noise_points,
        }