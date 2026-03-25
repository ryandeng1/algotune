import numpy as np
import hdbscan
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform robust clustering with outlier detection using HDBSCAN.

        Parameters
        ----------
        problem : dict
            Input dictionary containing keys:
                - "dataset" (array-like): data points.
                - "min_cluster_size" (int, optional): minimum cluster size.
                - "min_samples" (int, optional): minimum samples in a neighborhood.

        Returns
        -------
        dict
            Dictionary with keys:
                "labels" : list of cluster labels (-1 for noise).
                "probabilities" : list of cluster membership probabilities.
                "cluster_persistence" : list of cluster persistence values.
                "num_clusters" : integer count of clusters (excluding noise).
                "num_noise_points" : integer count of noise points.
        """
        # Convert dataset to numpy array
        dataset = np.asarray(problem["dataset"], dtype=float)

        # Default parameters if not provided
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Instantiate and fit HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=True
        )
        clusterer.fit(dataset)

        # Extract labels and probabilities
        labels = clusterer.labels_
        # Some older hdbscan versions may return None for probabilities; guard
        probabilities = (
            clusterer.probabilities_
            if hasattr(clusterer, "probabilities_") and clusterer.probabilities_ is not None
            else np.full_like(labels, fill_value=1.0, dtype=float)
        )
        persistence = (
            clusterer.cluster_persistence_
            if hasattr(clusterer, "cluster_persistence_") and clusterer.cluster_persistence_ is not None
            else np.full_like(labels, fill_value=0.0, dtype=float)
        )

        # Compute derived statistics
        num_clusters = int(len(set(labels[labels != -1])))
        num_noise_points = int(np.sum(labels == -1))

        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }

        return solution
