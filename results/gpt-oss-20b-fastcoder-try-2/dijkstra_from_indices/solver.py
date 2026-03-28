import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List, Union

class Solver:
    def __init__(self):
        self.directed = False          # undirected graph
        self.min_only = True           # return shortest distances only

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[Union[float, None]]]]:
        """
        Computes all‑pairs shortest distances from the given source indices.
        Results are returned as nested Python lists; unreachable nodes
        become ``None``.
        """
        # --- Parse the CSR representation ----------------------------------------------------
        try:
            # ``data``, ``indices``, ``indptr`` are expected to be list‑like
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            source_indices = problem["source_indices"]
            if not isinstance(source_indices, (list, tuple)) or not source_indices:
                raise ValueError("source_indices must be a non‑empty list")
        except Exception:
            return {"distances": []}

        # --- Compute shortest distances ------------------------------------------------------
        try:
            dist = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=source_indices,
                min_only=self.min_only,
            )
        except Exception:
            return {"distances": []}

        # --- Convert to pure Python list with ``None`` for infinities -------------------------
        if dist.ndim == 1:
            row = [None if np.isinf(x) else float(x) for x in dist]
            return {"distances": [row]}
        else:
            return {
                "distances": [
                    [None if np.isinf(x) else float(x) for x in row] for row in dist
                ]
            }