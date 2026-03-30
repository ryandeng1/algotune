from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    """
    Solver uses SciPy's Delaunay triangulation.
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"], dtype=np.float64)
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        return {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> list[tuple[int, int]]:
        """
        Return convex‑hull edges as sorted tuples (undirected),
        then sorted globally.
        """
        # ensure each edge is sorted internally
        sorted_edges = np.sort(edges, axis=1)
        # convert to list of tuples and sort the list
        return sorted(map(tuple, sorted_edges))

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> list[tuple[int, ...]]:
        """
        Return simplices as sorted tuples, entire list sorted.
        """
        # sort the indices within each simplex
        sorted_simplices = np.sort(simplices, axis=1)
        # convert to tuples and sort globally
        return sorted(map(tuple, sorted_simplices))