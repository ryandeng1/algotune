from typing import Any
import hdbscan
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the clustering problem using HDBSCAN.

        :param problem: A dictionary representing the clustering problem.
        :return: A dictionary with clustering solution details.
        """
        # Convert dataset to a NumPy array (efficient copy if already array)
        dataset = np.asarray(problem["dataset"])

        # Extract parameters with sensible defaults
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        clusterer.fit(dataset)

        # Gather clustering results
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Compute additional metrics
        noise_mask = labels == -1
        num_clusters = int(np.unique(labels[~noise_mask]).size)
        num_noise_points = int(np.sum(noise_mask))

        # Build solution dictionary
        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }
        return solution