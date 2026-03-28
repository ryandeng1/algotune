from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    """Fast Delaunay triangulation and convex hull extraction."""

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute all simplices and convex hull edges of the input point set.

        Parameters
        ----------
        problem : dict
            Must contain a key `'points'` mapping to an array‑like
            collection of (N, D) coordinates.

        Returns
        -------
        dict
            ``{'simplices': [...], 'convex_hull': [...]}``

            * ``simplices`` – list of sorted tuples of vertex indices
            * ``convex_hull`` – list of sorted tuples of edge endpoints
        """
        pts = np.asarray(problem["points"])
        # SciPy's Delaunay is already highly optimised (C + Fortran)
        tri = SciPyDelaunay(pts, incremental=False, qhull_options="Qt Qbb Qc Qz")
        simplices = tri.simplices
        convex_hull = tri.convex_hull

        return {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }

    @staticmethod
    def _canonical_edges(edges: np.ndarray) -> list[tuple[int, int]]:
        """
        Convert convex hull edges to canonical (sorted) tuples.

        Parameters
        ----------
        edges : (M, 2) array of integer indices

        Returns
        -------
        list[tuple[int, int]]
            List of sorted tuples, sorted globally for deterministic ordering.
        """
        # Sort each edge once and then convert to tuples
        sorted_edges = np.sort(edges, axis=1)
        return sorted(map(tuple, sorted_edges))

    @staticmethod
    def _canonical_simplices(simplices: np.ndarray) -> list[tuple[int, ...]]:
        """
        Convert simplices to canonical (sorted) tuples.

        Parameters
        ----------
        simplices : (K, D+1) array of integer indices

        Returns
        -------
        list[tuple[int, ...]]
            List of sorted tuples, sorted globally for deterministic ordering.
        """
        sorted_sips = np.sort(simplices, axis=1)
        return sorted(map(tuple, sorted_sips))