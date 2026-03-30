# solver.py
from __future__ import annotations
from typing import Any, List

import numpy as np
from sklearn.cluster import MiniBatchKMeans


class Solver:
    """
    A lightweight solver that performs K‑Means clustering on the input data.
    MiniBatchKMeans with n_init=1 is used to keep the runtime low while still
    providing reasonable cluster assignments.  If an exception occurs during
    fitting (e.g. invalid input data or k outside the valid range) the method
    returns a list of zeros of the same length as the input data.
    """

    def solve(self, problem: dict[str, Any]) -> List[int]:
        # Extract the data and number of clusters
        X = problem.get("X")
        k = problem.get("k", 0)

        # Quick sanity check – if k is not a positive integer, return zeros.
        if not isinstance(k, int) or k <= 0:
            return [0] * (len(X) if X is not None else 0)

        # Convert X to a NumPy array (handles lists, tuples, etc.)
        try:
            X_np = np.asarray(X, dtype=float)
            if X_np.ndim != 2:
                raise ValueError("X must be a 2‑D array")
        except Exception:
            return [0] * (len(X) if X is not None else 0)

        # If k is greater than the number of samples, cap it to the sample size.
        n_samples = X_np.shape[0]
        if k > n_samples:
            k = n_samples

        try:
            # Use MiniBatchKMeans with a single initialization to keep runtime low.
            nbk = MiniBatchKMeans(
                n_clusters=k,
                init="k-means++",
                n_init=1,   # single init – speeds up method drastically
                max_iter=300,
                batch_size=2048,
                random_state=0,
            )
            nbk.fit(X_np)
            return nbk.labels_.tolist()
        except Exception:
            # In case the fitting process crashes, fall back to zeros.
            return [0] * n_samples