from typing import Any

import numpy as np
from scipy.spatial import Delaunay as _SciPyDelaunay


def _canonical_simplices(simplices: np.ndarray) -> np.ndarray:
    """
    Return simplices sorted lexicographically and with vertex indices
    within each simplex sorted in ascending order.
    """
    # Sort the vertices inside each simplex
    simplices = np.sort(simplices, axis=1)
    # Convert to structured array to sort rows lexicographically
    dtype = [(f"f{i}", np.int64) for i in range(simplices.shape[1])]
    structured = np.array([tuple(row) for row in simplices], dtype=dtype)
    # Sort and recover as ndarray
    return np.array([tuple(row) for row in np.sort(structured)], dtype=np.int64)


def _canonical_edges(edges: np.ndarray) -> np.ndarray:
    """
    Return edges sorted lexicographically with each edge's vertices
    ordered in ascending order.
    """
    edges = np.sort(edges, axis=1)
    dtype = [(f"f{i}", np.int64) for i in range(edges.shape[1])]
    return np.array([tuple(row) for row in np.sort(np.array([tuple(row) for row in edges], dtype=dtype))],
                    dtype=np.int64)


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the Delaunay triangulation of a set of points and the convex hull
        edges. The result is returned with simplices and edges in a canonical
        (sorted) form that is independent of the input ordering.
        """
        # Convert points to a NumPy array; this ensures the input is float32/64
        pts = np.asarray(problem["points"], dtype=np.float64)

        # Perform the triangulation
        tri = _SciPyDelaunay(pts)
        simplices = tri.simplices
        hull = tri.convex_hull

        return {
            "simplices": _canonical_simplices(simplices),
            "convex_hull": _canonical_edges(hull),
        }