import numpy as np
from typing import Any
from scipy.spatial import Delaunay as SciPyDelaunay


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        return {
            "simplices": self._canonical_simplices(tri.simplices),
            "convex_hull": self._canonical_edges(tri.convex_hull),
        }

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> list[tuple[int, int]]:
        """Return undirected convex‑hull edges in canonical sorted order."""
        # sort vertices of each edge
        sorted_edges = np.sort(edges, axis=1)
        # determine lexicographic order of edges
        order = np.lexsort(sorted_edges.T[::-1])
        # convert to list of tuples
        return [tuple(row) for row in sorted_edges[order]]

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> list[tuple[int, ...]]:
        """Return simplices with vertices sorted within each simplex and sorted lexicographically."""
        # sort vertices inside each simplex
        sorted_simplices = np.sort(simplices, axis=1)
        # lexicographic ordering of simplices
        order = np.lexsort(sorted_simplices.T[::-1])
        return [tuple(row) for row in sorted_simplices[order]]