from typing import Any
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.min_only = True

    def solve(self, problem: dict[str, Any]) -> dict[str, list[list[float]]]:
        """
        Solve a single‑source or multi‑source shortest‑path problem on a CSR graph.
        Returns a nested list of distances (with ``None`` instead of ``np.inf``).
        """

        # Build CSR matrix once
        data = problem["data"]
        indices = problem["indices"]
        indptr = problem["indptr"]
        shape = problem["shape"]
        graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

        # Source nodes
        sources = np.asarray(problem["source_indices"], dtype=int)

        # Fast Dijkstra implementation
        dist = scipy.sparse.csgraph.dijkstra(
            csgraph=graph,
            directed=self.directed,
            indices=sources,
            min_only=self.min_only,
        )

        # Convert numpy array of shape (n,) or (k, n) to list of lists
        # Replace +inf with None
        if dist.ndim == 1:
            out = [[None if np.isinf(d) else d for d in dist]]
        else:
            out = [[None if np.isinf(d) else d for d in row] for row in dist]

        return {"distances": out}