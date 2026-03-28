from typing import Any
import numpy as np
import hdbscan

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the clustering problem using HDBSCAN.

        :param problem: A dictionary representing the clustering problem.
        :return: A dictionary with clustering solution details
        """
        # Convert input data to a NumPy array once – avoiding repeated Python
        # list conversions inside the dense HDBSCAN operations.
        dataset = np.array(problem['dataset'], dtype=float, copy=False)

        # Pull clustering hyper‑parameters with sensible defaults.
        min_cluster_size = problem.get('min_cluster_size', 5)
        min_samples = problem.get('min_samples', 3)

        # Instantiate and fit HDBSCAN directly on the NumPy array.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        clusterer.fit(dataset)

        # Extract the results.  Using NumPy's native operations keeps
        # the overhead minimal before converting to plain lists.
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Build the solution dictionary while keeping the data in list form
        # (required by the expected interface).
        # These conversions are inexpensive compared to the clustering
        # itself.
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels[unique_labels != -1])
        num_noise_points = int(np.sum(labels == -1))

        return {
            'labels': labels.tolist(),
            'probabilities': probabilities.tolist(),
            'cluster_persistence': persistence.tolist(),
            'num_clusters': num_clusters,
            'num_noise_points': num_noise_points
        }