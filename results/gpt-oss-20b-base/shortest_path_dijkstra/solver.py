from typing import Any, Dict
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

class Solver:
    method = "FW"          # Floyd‑Warshall, works for dense or sparse graphs
    directed = False

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes all-pairs shortest path distances.

        Parameters
        ----------
        problem : dict
            Dictionary with keys 'data', 'indices', 'indptr', and 'shape' describing a CSR matrix.

        Returns
        -------
        dict
            {'distance_matrix': list[list[float]]} with `None` where no path exists.
        """
        # Build CSR matrix in one go (fast)
        graph_csr = scipy.sparse.csr_matrix(
            (problem["data"], problem["indices"], problem["indptr"]),
            shape=problem["shape"],
        )

        # Compute distances
        dist_matrix = scipy.sparse.csgraph.shortest_path(
            csgraph=graph_csr,
            method=self.method,
            directed=self.directed,
        )

        # Convert infinities to None efficiently
        dist_matrix[dist_matrix == np.inf] = None

        # Convert to list of lists for the required output format
        return {"distance_matrix": dist_matrix.tolist()}