from typing import Any
import numpy as np
import hdbscan

class Solver:
    """
    Optimised HDBSCAN solver.

    The core‑logic remains the same but several HDBSCAN knobs are enabled to
    accelerate the clustering step:
      * Use all available CPU cores for the underlying
        nearest‑neighbour graph computation (`core_dist_n_jobs`),
        and for the single‑linkage tree construction (`connection_n_jobs`).
      * Increase the leaf size of the KD‑tree (`leaf_size`), which reduces
        the overhead for large datasets.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        # Ensure a contiguous float32 array for fast I/O
        dataset = np.asarray(problem['dataset'], dtype=np.float32, order='C')

        min_cluster_size = problem.get('min_cluster_size', 5)
        min_samples = problem.get('min_samples', 3)

        # Initialise HDBSCAN with performance‑oriented parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            core_dist_n_jobs=-1,        # parallel nearest‑neighbour
            connection_n_jobs=-1,      # parallel single‑linkage
            leaf_size=32,              # bigger leaf → fewer splits
        )
        clusterer.fit(dataset)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Convert NumPy arrays to plain Python lists for the required output format
        labels_list = labels.tolist()
        probabilities_list = probabilities.tolist()
        persistence_list = persistence.tolist()
        num_clusters = int(len(set(labels_list)) - (1 if -1 in labels_list else 0))
        num_noise_points = int((labels == -1).sum())

        return {
            'labels': labels_list,
            'probabilities': probabilities_list,
            'cluster_persistence': persistence_list,
            'num_clusters': num_clusters,
            'num_noise_points': num_noise_points
        }