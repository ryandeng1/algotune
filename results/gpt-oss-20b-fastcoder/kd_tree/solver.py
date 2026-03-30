# solver.py
from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import faiss


class Solver:
    """
    Fast k‑nearest neighbour solver using FAISS.

    The implementation avoids repeated type conversions, reduces memory copies,
    and vectorises the construction of the hyper‑cube boundary queries.
    """

    def __init__(self) -> None:
        # No heavy initialisation – the FAISS index is built per call
        self._index_cls = faiss.IndexFlatL2

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float32, copy=False)
        queries = np.asarray(problem["queries"], dtype=np.float32, copy=False)

        n_points, dim = points.shape
        k = int(problem["k"])
        k = k if k <= n_points else n_points

        # Build index
        index = faiss.IndexFlatL2(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(points, np.arange(n_points, dtype=np.int64))

        # Main search
        dist, idx = index.search(queries, k)
        solution: Dict[str, Any] = {
            "indices": idx.tolist(),
            "distances": dist.tolist(),
        }

        # Optional hyper‑cube shell boundary queries
        if problem.get("distribution") == "hypercube_shell":
            # One query per side of the hyper‑cube: 2 * dim
            # Initialise all‑zero and all‑one vectors
            zeros = np.zeros((2 * dim, dim), dtype=np.float32)
            ones = np.ones((2 * dim, dim), dtype=np.float32)

            # Set the appropriate entry to 0 for the zero‑vector side,
            # and 1 for the one‑vector side (already set)
            # the ith 0-vector and ith 1-vector share the same row index
            for d in range(dim):
                zeros[2 * d, d] = 0.0
                ones[2 * d, d] = 1.0
            # concatenate Zero and One queries into one matrix
            queries_hc = np.vstack((zeros, ones))

            bq_dist, bq_idx = index.search(queries_hc, k)
            solution["boundary_distances"] = bq_dist.tolist()
            solution["boundary_indices"] = bq_idx.tolist()

        return solution