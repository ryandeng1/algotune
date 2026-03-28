from typing import Any
import hdbscan
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        dataset = problem["dataset"]
        if not isinstance(dataset, np.ndarray):
            dataset = np.array(dataset)
        
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(dataset)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        has_noise = (labels == -1).any()
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if has_noise else 0)
        num_noise_points = int(np.sum(labels == -1))

        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }
        return solution