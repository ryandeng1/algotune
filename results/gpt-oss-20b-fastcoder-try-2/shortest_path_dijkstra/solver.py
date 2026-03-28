import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List

class Solver:
    """All‑pairs shortest–path solver with minimal overhead"""

    __slots__ = ("directed", "method")

    def __init__(self) -> None:
        self.directed: bool = False
        self.method: str = "D"  # Dijkstra (default)

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Computes the all‑pairs shortest‑path distance matrix for a graph
        supplied in CSR format.

        Parameters
        ----------
        problem : dict
            Must contain keys 'data', 'indices', 'indptr', and 'shape'.
            The values correspond to the three arrays that build a CSR matrix.

        Returns
        -------
        dict
            ``{"distance_matrix": [[...], ...]}`` where infinite distances are
            represented by ``None``.
        """
        try:
            csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            return {"distance_matrix": []}

        try:
            dist = scipy.sparse.csgraph.shortest_path(
                csr, directed=self.directed, method=self.method
            )
        except Exception:
            return {"distance_matrix": []}

        # Replace np.inf with None while keeping the array as an object
        # (to avoid unwanted numeric casts) and convert to nested lists.
        mask = np.isinf(dist)
        arr_obj = dist.astype(object)
        arr_obj[mask] = None

        return {"distance_matrix": arr_obj.tolist()}