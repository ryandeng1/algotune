import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self):
        self.directed = False
        self.min_only = True

    def solve(self, problem: dict) -> dict:
        """
        Solve shortest path distances from given source indices using scipy's
        sparse dijkstra implementation. Only distances are returned and
        non‑reachable nodes are represented as None.
        """
        # Build CSR matrix
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )

        source_indices = problem["source_indices"]
        if not source_indices:
            return {"distances": []}

        # Compute distances
        dist = scipy.sparse.csgraph.dijkstra(
            csgraph=graph_csr,
            directed=self.directed,
            indices=source_indices,
            min_only=self.min_only,
        )

        # Convert np.inf to None
        inf_mask = np.isinf(dist)
        dist = dist.tolist()
        for r, row in enumerate(dist):
            for c, val in enumerate(row):
                if inf_mask[r, c]:
                    row[c] = None

        return {"distances": dist}