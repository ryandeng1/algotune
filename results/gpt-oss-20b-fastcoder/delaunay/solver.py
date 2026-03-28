import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay
from typing import Any, Dict, List, Tuple

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        hull_edges = tri.convex_hull
        return {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(hull_edges),
        }

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> List[Tuple[int, int]]:
        # Canonicalise undirected edges: ensure lower index first and sort overall
        return sorted((min(a, b), max(a, b)) for a, b in edges)

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> List[Tuple[int, ...]]:
        # Canonicalise each simplex: sort vertex indices, then sort the whole list
        return sorted(tuple(sorted(s)) for s in simplices)