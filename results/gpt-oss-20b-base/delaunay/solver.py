import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay
from typing import Any, List, Tuple


def _canonical_simplices(simplices: np.ndarray) -> List[Tuple[int, ...]]:
    """Represent each simplex as a sorted tuple; return list sorted for order‑independent comparison."""
    return sorted(map(tuple, map(sorted, simplices)))


def _canonical_edges(edges: np.ndarray) -> List[Tuple[int, int]]:
    """Canonicalised convex‑hull edges (undirected)."""
    return sorted(map(tuple, map(sorted, edges)))


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Compute Delaunay triangulation and convex hull for a set of 2‑D points."""
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        return {
            "simplices": _canonical_simplices(simplices),
            "convex_hull": _canonical_edges(convex_hull),
        }
