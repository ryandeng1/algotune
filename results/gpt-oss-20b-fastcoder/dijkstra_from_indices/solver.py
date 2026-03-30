# solver.py

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from typing import Any, List


class Solver:
    """Optimised solver for shortest‑path distances using SciPy."""

    __slots__ = ("directed", "min_only")

    def __init__(self) -> None:
        # Adjacency matrix is undirected and only shortest distances are needed
        self.directed: bool = False
        self.min_only: bool = True

    def _prepare_csr(self, problem: dict[str, Any]) -> scipy.sparse.csr_matrix:
        """
        Create a CSR matrix from the problem dictionary.
        Raises ValueError if required keys are missing or malformed.
        """
        try:
            # We trust the format; avoid any heavy runtime checks
            data = problem["data"]
            indices = problem["indices"]
            indptr = problem["indptr"]
            shape = problem["shape"]
        except KeyError as exc:
            raise ValueError(f"Missing key in problem definition: {exc}") from None

        return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)

    @staticmethod
    def _source_indices(problem: dict[str, Any]) -> List[int]:
        src = problem.get("source_indices")
        if not isinstance(src, (list, tuple)) or not src:
            raise ValueError("source_indices must be a non‑empty list")
        return list(src)

    def _dist_to_lists(
        self, dist: np.ndarray
    ) -> List[List[float | None]]:
        """
        Convert a NumPy array of distances to a nested list
        with ``None`` in place of infinities.
        """
        # ``np.where`` keeps shape; we convert to lists afterwards
        inf_mask = np.isposinf(dist)
        if inf_mask.any():
            # Replace infinities by a placeholder; the placeholder can be any type
            # because we will convert afterwards
            dist = np.where(inf_mask, cast=object, x=dist, y=1)
            out = dist.tolist()
            for i, row in enumerate(out):
                for j, val in enumerate(row):
                    if val is 1:  # placeholder for inf
                        out[i][j] = None
        else:
            out = dist.tolist()
        return out

    def solve(self, problem: dict[str, Any]) -> dict[str, List[List[float | None]]]:
        """
        Compute shortest‑path distances from one or more source nodes.
        Returns a dictionary with key ``"distances"``.  The output is a list of lists;
        each inner list contains the distances from one source node.
        ``None`` represents an unreachable node.
        """
        try:
            graph = self._prepare_csr(problem)
            sources = self._source_indices(problem)
        except Exception:
            return {"distances": []}

        try:
            dist_arr = scipy.sparse.csgraph.dijkstra(
                csgraph=graph,
                directed=self.directed,
                indices=sources,
                min_only=self.min_only,
            )
        except Exception:
            return {"distances": []}

        # Convert the O(n*m) NumPy array to the required Python list
        if dist_arr.ndim == 1:
            # single source – shape (n,)
            out = [[None if np.isposinf(d) else d for d in dist_arr]]
        else:
            out = [[None if np.isposinf(d) else d for d in row] for row in dist_arr]

        return {"distances": out}