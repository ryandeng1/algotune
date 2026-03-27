from typing import Any, Dict, List
import numpy as np
import hdbscan


def solve(problem: Dict[str, Any]) -> Dict[str, List]:
    """
    Solve the clustering problem using HDBSCAN.

    Parameters
    ----------
    problem : dict
        Dictionary containing:
            - "dataset": 2‑D array‑like of shape (n_samples, n_features)
            - (optional) "min_cluster_size": int
            - (optional) "min_samples": int

    Returns
    -------
    dict
        Mapping with keys:
            - "labels": List[int] of cluster labels
            - "probabilities": List[float] of label probabilities
            - "cluster_persistence": List[float] of cluster persistence values
            - "num_clusters": int, number of clusters found (excl. noise)
            - "num_noise_points": int, number of points labelled as noise (-1)
    """
    # Convert dataset to a NumPy array once and share the memory
    dataset = np.asarray(problem["dataset"], dtype=float)

    # Pull hyper‑parameters with sane defaults
    min_cluster_size = int(problem.get("min_cluster_size", 5))
    min_samples = int(problem.get("min_samples", 3))

    # Instantiate and fit the HDBSCAN model
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    clusterer.fit(dataset)

    # Extract results in the fastest possible manner
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    persistence = clusterer.cluster_persistence_

    # Count clusters and noise via NumPy for speed
    non_noise = labels != -1
    # np.unique is faster than a Python set for large arrays
    num_clusters = int(np.unique(labels[non_noise]).size if non_noise.any() else 0)
    num_noise_points = int(np.count_nonzero(labels == -1))

    # Build the solution with minimal Python overhead
    solution = {
        "labels": labels.tolist(),
        "probabilities": probabilities.tolist(),
        "cluster_persistence": persistence.tolist(),
        "num_clusters": num_clusters,
        "num_noise_points": num_noise_points,
    }
    return solution