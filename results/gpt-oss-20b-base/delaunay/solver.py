from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem['points'])
        tri = SciPyDelaunay(pts)
        result = {
            'simplices': self._canonical_simplices(tri.simplices),
            'convex_hull': self._canonical_edges(tri.convex_hull)
        }
        return result

    def _canonical_edges(self, edges: np.ndarray) -> list[tuple[int, int]]:
        # Undirected edges in canonical order
        return sorted(map(lambda e: tuple(sorted(e)), edges))

    def _canonical_simplices(self, simplices: np.ndarray) -> list[tuple[int, ...]]:
        # Simplex vertices in canonical order, sorted list for deterministic output
        return sorted(map(lambda s: tuple(sorted(s)), simplices))