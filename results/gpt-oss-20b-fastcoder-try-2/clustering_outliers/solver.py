from typing import Any
import hdbscan
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """Cluster dataset with hdbscan as fast as possible."""
        # Convert input to a NumPy array once
        X = np.asarray(problem['dataset'], dtype=float)

        # Pull hyper‑parameters, falling back to sensible defaults
        min_cluster_size = problem.get('min_cluster_size', 5)
        min_samples = problem.get('min_samples', 3)

        # Initialise and fit the HDBSCAN model
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            algorithm="generic",
            leaf_size=40,          # default leaf size for speed
            copy=False,
            cluster_selection_method="eom",
        )
        clusterer.fit(X)

        # Extract labels and probabilities
        labels = clusterer.labels_
        probs = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Quick statistics
        noise_mask = labels == -1
        num_clusters = int(len(set(labels[~noise_mask])))
        num_noise = int(np.count_nonzero(noise_mask))

        # Convert everything to plain Python lists
        solution = {
            'labels': labels.tolist(),
            'probabilities': probs.tolist(),
            'cluster_persistence': persistence.tolist(),
            'num_clusters': num_clusters,
            'num_noise_points': num_noise,
        }
        return solution