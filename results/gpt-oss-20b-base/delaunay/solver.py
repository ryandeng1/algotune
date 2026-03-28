from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> list[tuple[int, int]]:
        """Return undirected convex-hull edges in canonical (sorted) form."""
        # sort each edge locally
        edges_sorted = np.sort(edges, axis=1)
        # convert to list of tuples and sort globally
        return sorted(map(tuple, edges_sorted))

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> list[tuple[int, ...]]:
        """Return Delaunay simplices as sorted tuples, sorted globally."""
        # sort vertex indices in each simplex
        simplices_sorted = np.sort(simplices, axis=1)
        # convert to tuples
        tuples = [tuple(row) for row in simplices_sorted]
        # sort the list of tuples
        return sorted(tuples)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem['points'])
        tri = SciPyDelaunay(pts)
        result = {
            'simplices': self._canonical_simplices(tri.simplices),
            'convex_hull': self._canonical_edges(tri.convex_hull),
        }
        return result