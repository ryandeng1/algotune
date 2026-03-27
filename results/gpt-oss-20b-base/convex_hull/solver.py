from typing import Any, Dict, List
import numpy as np
from scipy.spatial import ConvexHull


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convex hull of a set of points using SciPy's
        ConvexHull implementation.

        Parameters
        ----------
        problem : dict
            Dictionary containing the key ``"points"``, which is an
            (n, d) array-like of point coordinates.

        Returns
        -------
        dict
            Dictionary with two keys:
            * ``"hull_vertices"`` – a list of the indices of the input
              points that form the hull.
            * ``"hull_points"`` – a list of the corresponding coordinate
              values in the same order as ``hull_vertices``.
        """
        # Ensure input is a NumPy array for fast indexing
        points = np.asarray(problem["points"])

        # Compute the convex hull
        hull = ConvexHull(points)

        # Convert the indices and points to plain Python lists
        hull_vertices: List[int] = hull.vertices.tolist()
        hull_points: List[List[float]] = points[hull.vertices].tolist()

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}