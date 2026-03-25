# solver.py
import hdbscan
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Perform robust clustering with HDBSCAN and return required fields.
        """
        # Extract data
        dataset = np.array(problem["dataset"])
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples)
        clusterer.fit(dataset)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Compute derived counts
        num_clusters = len(set(labels[labels != -1]))
        num_noise = int(np.sum(labels == -1))

        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise
        }
