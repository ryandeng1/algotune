import numpy as np
import hdbscan
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Fit an HDBSCAN clustering model and return key results.

        Parameters
        ----------
        problem : dict
            Dictionary containing the dataset and optional parameters
            for the clustering algorithm.

        Returns
        -------
        dict
            Contains cluster labels, per-point probabilities,
            cluster persistence, count of clusters and noise points.
        """
        # Convert the input dataset to a NumPy array of the most efficient
        # dtype for hdbscan.  The default dtype (float64) is typically the
        # fastest for numerical work in C extensions.
        dataset = np.asarray(problem["dataset"], dtype=np.float64)

        # Pull hyper‑parameters, falling back to HDBSCAN defaults if missing.
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Run HDBSCAN once.  Instantiating and fitting in a single call
        # reduces Python‑level overhead compared to separate steps.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        clusterer.fit(dataset)

        # Extract results
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Compute cluster statistics directly on the NumPy arrays
        # (no Python loops).
        num_clusters = int(np.count_nonzero(labels != -1))
        num_noise_points = int(np.sum(labels == -1))

        # Convert NumPy arrays to Python lists for JSON serialisation
        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }