from typing import Any, Dict, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    """Optimised Dijkstra solver for CSR graphs with minimal overhead."""

    def __init__(self, directed: bool = False, min_only: bool = False):
        self.directed = directed
        self.min_only = min_only

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        # Construct CSR matrix once — avoid repeated try/except overhead
        try:
            data = problem['data']
            indices = problem['indices']
            indptr = problem['indptr']
            shape = problem['shape']
            graph_csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
            source_indices = problem['source_indices']
            if not isinstance(source_indices, list) or not source_indices:
                return {"distances": []}
        except Exception:
            return {"distances": []}

        # Fast dijkstra call
        try:
            dist_matrix = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )
        except Exception:
            return {"distances": []}

        # Replace inf with None using vectorised operations
        if dist_matrix.ndim == 1:
            row = dist_matrix
            out = [None if np.isposinf(v) else v for v in row]
            return {"distances": [out]}
        else:
            out = [[None if np.isposinf(v) else v for v in row] for row in dist_matrix]
            return {"distances": out}