from typing import Any, Dict, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    def __init__(self, directed: bool = True, min_only: bool = False):
        self.directed = directed
        self.min_only = min_only

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Computes shortest‑path distances from the given source nodes using
        scipy.sparse.csgraph.dijkstra and returns a nested list with
        ``None`` in place of infinite distances.
        """
        data, indices, indptr = problem['data'], problem['indices'], problem['indptr']
        shape = problem['shape']
        graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

        src = problem['source_indices']
        if not src:
            return {"distances": []}

        # Run Dijkstra's algorithm
        dist = scipy.sparse.csgraph.dijkstra(
            csgraph=graph_csr,
            directed=self.directed,
            indices=src,
            min_only=self.min_only
        )

        # Convert infinities to None and cast to list of lists
        if dist.ndim == 1:          # single source
            row = [None if np.isinf(v) else float(v) for v in dist]
            return {"distances": [row]}

        rows = []
        for r in dist:
            rows.append([None if np.isinf(v) else float(v) for v in r])
        return {"distances": rows}