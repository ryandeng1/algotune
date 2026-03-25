import numpy as np
from numba import njit

@njit
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

@njit
def convex_hull(points):
    n = len(points)
    if n <= 1:
        return points
    lower = []
    for i in range(n):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], points[i]) <= 0:
            lower.pop()
        lower.append(points[i])
    upper = []
    for i in range(n-1, -1, -1):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], points[i]) <= 0:
            upper.pop()
        upper.append(points[i])
    return lower[:-1] + upper[:-1]

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = problem["points"]
        n = problem["n"]
        points_with_index = [(p[0], p[1], i) for i in range(n)]
        points_with_index.sort(key=lambda x: (x[0], x[1]))
        hull_points = convex_hull(points_with_index)
        hull_vertices = [p[2] for p in hull_points]
        hull_points = [points[i] for i in hull_vertices]
        return {
            "hull_vertices": hull_vertices,
            "hull_points": hull_points
        }
