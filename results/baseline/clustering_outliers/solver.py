from typing import Any
import hdbscan
import numpy as np

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the clustering problem using HDBSCAN.

        :param problem: A dictionary representing the clustering problem.
        :return: A dictionary with clustering solution details
        """
        dataset = np.array(problem['dataset'])
        min_cluster_size = problem.get('min_cluster_size', 5)
        min_samples = problem.get('min_samples', 3)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(dataset)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        solution = {'labels': labels.tolist(), 'probabilities': probabilities.tolist(), 'cluster_persistence': persistence.tolist(), 'num_clusters': len(set(labels[labels != -1])), 'num_noise_points': int(np.sum(labels == -1))}
        return solution
