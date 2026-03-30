# solver.py
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List


class Solver:
    """
    Optimised shortest‑path solver using scipy's sparse dijkstra.
    """

    __slots__ = ("directed", "min_only")

    def __init__(self) -> None:
        # Default behaviour matches the original implementation
        self.directed: bool = False
        self.min_only: bool = True

    @staticmethod
    def _to_list_from_ndarray(arr: np.ndarray) -> List[List[float | None]]:
        """
        Convert a 1‑D or 2‑D numpy array of distances to a nested list
        replacing np.inf with None.
        """
        if arr.ndim == 1:
            return [[None if np.isinf(x) else float(x) for x in arr]]
        # Convert each row
        return [[None if np.isinf(x) else float(x) for x in row] for row in arr]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float | None]]]:
        """
        Shortest‑path distances from the provided source indices.
        """
        try:
            graph_csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
            sources: List[int] = problem["source_indices"]
            if not isinstance(sources, list) or not sources:
                raise ValueError
        except Exception:
            return {"distances": []}

        try:
            dist = scipy.sparse.csgraph.dijkstra(
                csgraph=graph_csr,
                directed=self.directed,
                indices=sources,
                min_only=self.min_only,
            )
        except Exception:
            return {"distances": []}

        return {"distances": self._to_list_from_ndarray(dist)}