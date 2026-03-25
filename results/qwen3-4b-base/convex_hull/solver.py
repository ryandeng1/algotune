import numpy as np
from scipy.spatial import ConvexHull

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.array(problem["points"], dtype=np.float64)
        hull = ConvexHull(points)
        hull_vertices = hull.vertices.tolist()
        hull_points = points[hull.vertices].tolist()
        return {
            "hull_vertices": hull_vertices,
            "hull_points": hull_points
        }
