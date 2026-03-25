import numpy as np
import hdbscan

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        dataset = np.array(problem["dataset"], dtype=np.float32)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        clusterer.fit(dataset)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        
        clusters = labels[labels != -1]
        num_clusters = len(np.unique(clusters))
        num_noise_points = int(np.sum(labels == -1))
        
        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points
        }
