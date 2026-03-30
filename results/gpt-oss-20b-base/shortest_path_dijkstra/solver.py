from typing import Any, Dict, List
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    # Use a dense output for speed; therefore keep directed = False
    # and method = 'D' (Dijkstra for sparse graphs) as the default.
    def __init__(self):
        self.directed = False
        self.method = 'D'

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Compute all‑pairs shortest paths for a graph given in CSR components.
        The result is returned as a list of lists with `None` for unreachable nodes.
        """
        # Build sparse CSR matrix directly from the provided components.
        graph_csr = scipy.sparse.csr_matrix(
            (problem['data'], problem['indices'], problem['indptr']),
            shape=problem['shape']
        )

        # Run the shortest‑path algorithm (single source Dijkstra expanded to all pairs).
        dist_matrix = scipy.sparse.csgraph.shortest_path(
            csgraph=graph_csr,
            method=self.method,
            directed=self.directed
        )

        # Replace inf with None and convert to a Python list of lists.
        # We convert to object dtype first to allow None entries.
        mask = np.isinf(dist_matrix)
        if mask.any():
            obj_matrix = dist_matrix.astype(object)
            obj_matrix[mask] = None
        else:
            # All entries are finite – no need to create an object array.
            obj_matrix = dist_matrix.astype(object)

        return {"distance_matrix": obj_matrix.tolist()}