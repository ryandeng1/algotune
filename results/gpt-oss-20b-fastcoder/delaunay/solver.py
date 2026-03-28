from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        return {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> list[tuple[int, int]]:
        """Return convex‑hull edges in canonical form (sorted tuples)."""
        return sorted(tuple(sorted(e)) for e in edges)

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> list[tuple[int, ...]]:
        """Return Delaunay simplices in canonical form (each simplex as a sorted tuple)."""
        return sorted(tuple(sorted(s)) for s in simplices)