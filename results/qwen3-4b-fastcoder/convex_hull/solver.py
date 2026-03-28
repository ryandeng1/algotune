from typing import Any
from scipy.spatial import ConvexHull


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = problem["points"]
        hull = ConvexHull(points)
        vertices = hull.vertices
        hull_vertices = vertices.tolist()
        hull_points = points[vertices].tolist()
        return {"hull_vertices": hull_vertices, "hull_points": hull_points}