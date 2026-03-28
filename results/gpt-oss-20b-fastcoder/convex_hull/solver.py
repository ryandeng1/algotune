import numpy as np
from scipy.spatial import ConvexHull
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convex hull of the given set of points using SciPy's ConvexHull.
        The points are expected to be in a format compatible with NumPy arrays.

        :param problem: A dictionary with a key 'points' containing the coordinates.
        :return: A dictionary containing the indices of the hull vertices and the coordinates
                 of those vertices.
        """
        # Convert input to a NumPy array (fastest handling by SciPy)
        points = np.asarray(problem['points'])
        # Compute the convex hull
        hull = ConvexHull(points)
        # Extract vertex indices and coordinates
        hull_vertices = hull.vertices.tolist()
        hull_points = points[hull.vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}