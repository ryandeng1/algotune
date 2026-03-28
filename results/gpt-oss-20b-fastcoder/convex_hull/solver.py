import numpy as np
from scipy.spatial import ConvexHull
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the convex hull of a set of points using scipy.spatial.ConvexHull.
        Fast conversion to NumPy array ensures the underlying C routine is used.
        """
        # Ensure we have a NumPy array (passed value may be list-like)
        pts = np.asarray(problem["points"], dtype=float)

        # Compute convex hull
        hull = ConvexHull(pts)

        # Convert the hull indices and points to Python lists
        hull_vertices = hull.vertices.tolist()
        hull_points = pts[hull.vertices].tolist()

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}