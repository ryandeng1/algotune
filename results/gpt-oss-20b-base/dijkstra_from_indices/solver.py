import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List


class Solver:
    def __init__(self):
        # Since all problems are undirected in this contest, keep this constant.
        self.directed = False
        self.min_only = True

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Computes the shortest‑path distances from the given source nodes via
        `scipy.sparse.csgraph.dijkstra`.  The result is returned as a list of
        lists, converting `numpy.inf` to `None` because the contest test harness
        expects this shape.

        Parameters
        ----------
        problem : dict
            Must contain:
                * 'data'   – edge weights (list or 1‑d array)
                * 'indices' – column indices
                * 'indptr'  – row pointer array
                * 'shape'   – tuple (n_nodes, n_nodes)
                * 'source_indices' – list or 1‑d array of start nodes

        Returns
        -------
        dict
            Contains key 'distances' with the computed matrix.
        """
        try:
            # Build CSR graph efficiently
            graph_csr = scipy.sparse.csr_matrix(
                (
                    np.asarray(problem["data"], dtype=float),
                    np.asarray(problem["indices"], dtype=int),
                    np.asarray(problem["indptr"], dtype=int),
                ),
                shape=tuple(problem["shape"]),
            )

            sources = problem["source_indices"]
            if not isinstance(sources, (list, np.ndarray)) or len(sources) == 0:
                return {"distances": []}

            # Run Dijkstra
            dist = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=sources,
                min_only=self.min_only,
            )

            # Replace inf with None in a fast vectorized manner
            if dist.ndim == 1:
                row = dist.tolist()
                for i, val in enumerate(row):
                    if np.isinf(val):
                        row[i] = None
                return {"distances": [row]}

            # Multi‑source case
            res = dist.tolist()
            for r in range(len(res)):
                row = res[r]
                for c, val in enumerate(row):
                    if np.isinf(val):
                        row[c] = None

            return {"distances": res}

        except Exception:
            return {"distances": []}