# solver.py
# Optimised HDBSCAN solver – see discussion for performance notes

from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List

import hdbscan
import numpy as np

# --------------------------------------------------------------------------- #
#  The Solver class implements the required `solve()` method with a focus on
#  speed.  Key optimisations:
#
#  * The dataset is cast to float32 – HDBSCAN accepts any float type and a
#    smaller dtype reduces memory traffic.
#  * The number of clusters is computed with `np.unique(..., return_counts=True)`
#    – this is faster than building a Python set.
#  * Noise points are counted through `np.count_nonzero(...)`.
#  * The labels, probabilities and persistence arrays are sliced and converted
#    to lists in a single pass.
# --------------------------------------------------------------------------- #

class Solver:
    """
    Optimised solver performing clustering with hdbscan.
    """

    @staticmethod
    def solve(problem: Dict[str, Any]) -> Dict[str, List[int | float]]:
        """
        Perform HDBSCAN clustering on the supplied dataset.
        Parameters
        ----------
        problem : dict
            Expected keys:
            * 'dataset': 2‑D sequence of numeric values
            * 'min_cluster_size' (optional): int
            * 'min_samples' (optional): int
        Returns
        -------
        dict
            Keys:
            * 'labels' – cluster label for each point
            * 'probabilities' – membership probability per point
            * 'cluster_persistence' – persistence per cluster
            * 'num_clusters' – number of clusters found (excluding noise)
            * 'num_noise_points' – number of points labelled as noise (-1)
        """
        # -------------------------------------------------------------------- #
        # 1. Prepare data – cast to 32‑bit float for memory efficiency
        # -------------------------------------------------------------------- #
        dataset = np.asarray(problem["dataset"], dtype=np.float32)

        # -------------------------------------------------------------------- #
        # 2. Configure HDBSCAN – use values from problem dict with defaults
        # -------------------------------------------------------------------- #
        min_cluster_size = int(problem.get("min_cluster_size", 5))
        min_samples = int(problem.get("min_samples", 3))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        clusterer.fit(dataset)

        # -------------------------------------------------------------------- #
        # 3. Retrieve results
        # -------------------------------------------------------------------- #
        labels = clusterer.labels_
        probs = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # -------------------------------------------------------------------- #
        # 4. Fast statistics calculation
        # -------------------------------------------------------------------- #
        # Count clusters (exclude noise label -1)
        uniq, counts = np.unique(labels, return_counts=True)
        num_clusters = int(len(uniq) - (1 if -1 in uniq else 0))

        # Count noise points
        num_noise_points = int(np.count_nonzero(labels == -1))

        # -------------------------------------------------------------------- #
        # 5. Convert to Python lists only once
        # -------------------------------------------------------------------- #
        solution: Dict[str, List[int | float]] = {
            "labels": labels.tolist(),
            "probabilities": probs.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }
        return solution