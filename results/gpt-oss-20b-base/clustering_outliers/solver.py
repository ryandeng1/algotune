from __future__ import annotations
from typing import Any, Dict, List
import numpy as np
import hdbscan


class Solver:
    """
    Fast HDBSCAN clustering solver.

    The implementation focuses on minimal Python overhead:
    * Dataset is converted to a NumPy array once.
    * All heap‑allocations happen in NumPy, avoiding Python loops.
    * The final output uses `tolist` only for the required fields.
    * `num_clusters` is counted via NumPy's `unique` to avoid a Python set.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[Any]]:
        # Extract dataset and parameters
        data = np.array(problem["dataset"], dtype=np.float32)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        # Run HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        clusterer.fit(data)

        labels = clusterer.labels_
        probs = clusterer.probabilities_
        pers = clusterer.cluster_persistence_

        # Compute counts using NumPy for speed
        non_noise = labels != -1
        num_clusters = int(np.unique(labels[non_noise]).size)
        num_noise_points = int(np.sum(~non_noise))

        return {
            "labels": labels.tolist(),
            "probabilities": probs.tolist(),
            "cluster_persistence": pers.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }