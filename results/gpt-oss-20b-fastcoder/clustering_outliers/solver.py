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
        # Convert to a NumPy array with the most efficient dtype
        dataset = np.asarray(problem["dataset"], dtype=np.float32)

        # Pull algorithm parameters from the problem dict (fall back to defaults)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Perform clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        clusterer.fit(dataset)

        # Extract results
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Calculate cluster statistics with NumPy for speed
        unique, counts = np.unique(labels, return_counts=True)
        num_clusters = int((unique != -1).sum())
        num_noise_points = int((labels == -1).sum())

        # Build the solution dictionary
        solution: dict[str, list] = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }

        return solution