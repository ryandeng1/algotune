# solver.py
from typing import Any
import numpy as np
from scipy.spatial import ConvexHull


class Solver:
    """
    Computes the convex hull of a set of 2‑D or 3‑D points using
    :class:`scipy.spatial.ConvexHull`.  The implementation is focused on
    speed: the input data is converted once to a ``numpy.ndarray`` and
    the expensive ``.tolist()`` conversions are performed only once.
    """

    def __init__(self) -> None:
        # Cache the ConvexHull constructor and attributes to avoid global lookups
        self._ConvexHull = ConvexHull

    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the convex hull of the given set of points.

        Parameters
        ----------
        problem
            Dictionary with a key ``"points"`` pointing to an iterable of
            coordinate tuples or a ``numpy.ndarray``.

        Returns
        -------
        dict
            A dictionary containing:
            * ``"hull_vertices"`` – list of integer indices of the hull vertices
            * ``"hull_points"``   – list of the corresponding point coordinates
        """
        points = np.asarray(problem["points"])
        hull = self._ConvexHull(points)
        # ``hull.vertices`` is a 1‑D array of integer indices
        hull_vertices = hull.vertices.tolist()
        # Slice once and convert to list of coordinates
        hull_points = points[hull.vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}