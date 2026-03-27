from typing import Any
import numpy as np
from scipy.spatial import Delaunay

class Solver:
    """
    Optimised solver for Delaunay triangulation and convex hull extraction.

    The function keeps the public API unchanged but reduces memory traffic
    by:
    • Avoiding unnecessary copies when the input is already a NumPy array.
    • Using int32 arrays for indices, which are the minimal type that can
      represent typical problem sizes.
    • Pre‑allocating the final output dictionary to avoid repeated
      dynamic growth.
    """
    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> np.ndarray:
        """Return simplices sorted top‑level and then lexicographically."""
        return np.sort(np.sort(simplices, axis=1), axis=0).T

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> np.ndarray:
        """Return edges sorted lexicographically."""
        return np.sort(np.sort(edges, axis=1), axis=0)

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Ensure we work with the minimal required dtype
        pts = np.asarray(problem["points"], dtype=np.float64)  # keeps original if possible

        # Perform Delaunay triangulation
        tri = Delaunay(pts, incremental=False, qhull_options="Qt")

        # Retrieve integer indices in the smallest suitable dtype
        simplices = tri.simplices.astype(np.int32, copy=False)
        convex_hull = tri.convex_hull.astype(np.int32, copy=False)

        return {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }