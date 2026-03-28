import hdbscan
import numpy as np
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Run HDBSCAN clustering on the supplied dataset.

        Parameters
        ----------
        problem : dict
            Must contain a key ``'dataset'`` with an array‑like structure.
            Optional keys are ``'min_cluster_size'`` (default 5) and
            ``'min_samples'`` (default 3).

        Returns
        -------
        dict
            Contains:
            * ``labels`` – cluster assignments, ``-1`` for noise
            * ``probabilities`` – cluster membership probability
            * ``cluster_persistence`` – persistence of each cluster
            * ``num_clusters`` – number of positive clusters
            * ``num_noise_points`` – number of points marked as noise
        """
        # Convert input dataset to a NumPy array at once.
        data = np.asarray(problem["dataset"], dtype=float)

        # Extract optional HDBSCAN parameters, falling back to the defaults.
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Train HDBSCAN.
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            prediction_data=True,  # keep probability information
        )
        clusterer.fit(data)

        # Pull the results from the estimator.
        labels = clusterer.labels_.astype(int)
        probs = clusterer.probabilities_.astype(float)
        persistence = clusterer.cluster_persistence_.astype(float)

        # Compute statistics using vectorised operations.
        num_clusters = int(np.count_nonzero(labels != -1))
        num_noise = int(np.count_nonzero(labels == -1))

        # Return all results as plain Python lists for compatibility.
        return {
            "labels": labels.tolist(),
            "probabilities": probs.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise,
        }