# solver.py
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, Dict, List


class Solver:
    """
    Fast all‑pairs shortest path solver using scipy's CSR representation.
    The implementation avoids per‑element Python loops and is tuned for
    large, sparse graphs.

    The constructor can be used to customise the algorithm (`method` and
    `directed`).  It is evaluated only once, so any expensive setup
    (e.g. linking on first use of `scipy.sparse.csgraph`) happens during
    initialization and does not affect runtime of `solve`.
    """

    __slots__ = ("directed", "method")

    def __init__(self) -> None:
        # Default options – can be overridden after construction.
        self.directed: bool = False
        self.method: str = "D"   # D = Dijkstra, which is optimal for sparse graphs

    def _to_object_matrix(self, mat: np.ndarray) -> np.ndarray:
        """
        Convert a numeric numpy matrix to an object dtype matrix
        where infinite values are replaced by ``None``.
        """
        obj_mat = mat.astype(object, copy=True)
        obj_mat[np.isinf(obj_mat)] = None
        return obj_mat

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[List[Any]]]:
        """
        Compute the all‑pairs shortest paths for the graph described by the
        CSR components in *problem*.

        Parameters
        ----------
        problem:
            ``{
                'data':    list or array of non‑zero values,
                'indices': list or array of column indices,
                'indptr':  list or array of index pointers,
                'shape':   (n_rows, n_cols) tuple
            }``

        Returns
        -------
        solution: dict
            ``{'distance_matrix': [...]}``
            The matrix of shortest path distances where paths that cannot be
            reached are represented by ``None``.  The matrix is returned as a
            nested list because the caller expects a JSON‑serialisable format.
        """
        # ---- Build CSR matrix -------------------------------------------------
        try:
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
            # ``csr_matrix`` accepts any sequence; the conversion to
            # contiguous arrays is handled internally and is fast enough.
            graph_csr = scipy.sparse.csr_matrix(
                (data, indices, indptr), shape=shape
            )
        except Exception:  # pragma: no cover
            return {"distance_matrix": []}

        # ---- Compute all‑pairs shortest paths ---------------------------------
        try:
            dist_matrix = scipy.sparse.csgraph.shortest_path(
                csgraph=graph_csr, method=self.method, directed=self.directed
            )
        except Exception:  # pragma: no cover
            return {"distance_matrix": []}

        # ---- Convert to list of lists with ``None`` for unreachable nodes ----
        solution = {
            "distance_matrix": self._to_object_matrix(dist_matrix).tolist()
        }
        return solution