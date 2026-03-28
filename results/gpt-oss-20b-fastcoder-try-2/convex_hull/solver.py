from typing import Any
import numpy as np
from scipy.spatial import ConvexHull

class Solver:
    """
    Fast convex–hull solver.

    The implementation relies on Scipy's highly optimised Qhull backend.  
    No additional Python loops are used, which keeps the pure‑Python
    overhead minimal.
    """
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        hull = ConvexHull(points)
        # Convert to python lists for the required output format
        hull_vertices = hull.vertices.tolist()
        hull_points = points[hull.vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}