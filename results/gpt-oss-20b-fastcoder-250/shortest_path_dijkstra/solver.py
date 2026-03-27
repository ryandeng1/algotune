import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List


class Solver:
    """
    Optimised all‑pairs shortest‑path solver.

    * Builds CSR matrix in a single call.
    * Uses the most efficient SciPy shortest_path implementation.
    * Replaces infinite distances with ``None`` in a vectorised way.
    """

    def __init__(self, method: str = "Dijkstra", directed: bool = True):
        """
        Parameters
        ----------
        method : str, optional
            Algorithm used by :func:`scipy.sparse.csgraph.shortest_path`.
            Default is 'Dijkstra'.
        directed : bool, optional
            Whether the graph is directed or not.
        """
        self.method = method
        self.directed = directed

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[float]]]:
        """
        Solve the all‑pairs shortest path problem.

        Parameters
        ----------
        problem : dict
            A CSR representation of a weighted graph with keys:
                - 'data': array of non‑zero values
                - 'indices': column indices of the non‑zero values
                - 'indptr': index pointers for row start
                - 'shape': 2‑tuple of the matrix shape

        Returns
        -------
        dict
            ``{'distance_matrix': [[...], ...]}`` where `None` indicates that
            two nodes are disconnected.
        """
        try:
            csr = scipy.sparse.csr_matrix(
                (problem["data"], problem["indices"], problem["indptr"]),
                shape=problem["shape"],
            )
        except Exception:
            # If CSR construction fails return an empty matrix
            return {"distance_matrix": []}

        try:
            dist = scipy.sparse.csgraph.shortest_path(
                csgraph=csr, method=self.method, directed=self.directed
            )
        except Exception:
            return {"distance_matrix": []}

        # Replace infinities with None in a vectorised manner
        if np.any(np.isinf(dist)):
            dist = np.where(np.isinf(dist), np.nan, dist)
            # ``np.nan`` will be turned into ``None`` by the ``tolist`` conversion
        else:
            # No need for replacement – fast path
            pass

        # Convert the numpy array to a list of lists.
        # ``tolist`` keeps ``float('nan')`` as ``nan`` which JSON encodes as ``null``.
        distance_matrix = dist.tolist()

        return {"distance_matrix": distance_matrix}