import numpy as np
from scipy.spatial import ConvexHull

class Solver:
    """
    Optimized implementation of convex hull computation.
    The main optimization lies in minimal data conversion:
    * `points` is expected to be a NumPy array; if not, it is converted once.
    * The hull vertices are obtained directly as a flat list.
    """

    def _ensure_numpy(self, points):
        """Convert input to a 2-D NumPy array if necessary."""
        if isinstance(points, np.ndarray):
            return points
        return np.asarray(points, dtype=np.float64)

    def solve(self, problem: dict) -> dict:
        points = self._ensure_numpy(problem["points"])
        hull = ConvexHull(points)
        # Convert array indices to plain Python ints for serialization friendliness
        hull_vertices = hull.vertices.tolist()
        hull_points = points[hull.vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}