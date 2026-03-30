# solver.py
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay
from typing import Any, Dict, List, Tuple


class Solver:
    """
    A fast wrapper around SciPy's Delaunay triangulation.
    The result contains all simplices and the convex‑hull edges
    in canonical (sorted) order.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the Delaunay triangulation of the given points and
        return canonical representations for simplices and convex‑hull
        edges.  All operations are vectorised using NumPy for maximum
        speed.

        Parameters
        ----------
        problem : dict
            Expected to contain a key ``"points"`` whose value is
            any array‑like representation of shape (N, D).

        Returns
        -------
        dict
            ``{"simplices": [...], "convex_hull": [...]}``
            where both lists are sorted tuples of indices.
        """
        pts = np.asarray(problem["points"], dtype=np.int64, order="C")
        tri = SciPyDelaunay(pts)

        # Canonicalise simplices (triangles/tetrahedra)
        simplices = tri.simplices
        verbs = self._canonical_rows(simplices)

        # Canonicalise convex‐hull edges (undirected pairs)
        edges = tri.convex_hull
        hull_edges = self._canonical_rows(edges)

        return {"simplices": verbs, "convex_hull": hull_edges}

    @staticmethod
    def _canonical_rows(rows: np.ndarray) -> List[Tuple[int, ...]]:
        """
        Sort each row internally and then lexicographically sort all rows.
        The result is a list of tuples ready for dictionary comparisons.
        """
        # Ensure integer type
        rows = rows.astype(np.int64, copy=False)
        # Sort each row in place (fast C routine)
        rows.sort(axis=1)
        # Lexicographic sorting of the whole array
        # lexsort expects keys from the last to the first
        keys = rows.T[::-1]
        order = np.lexsort(keys)  # 1‑D indices
        sorted_rows = rows[order]
        # Convert to list of tuples
        return [tuple(row) for row in sorted_rows]