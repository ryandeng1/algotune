from typing import Any, Dict, List, Tuple

class Solver:

    @staticmethod
    def _convex_hull(points: List[Tuple[float, float]]) -> List[int]:
        """
        Computes the convex hull of a set of 2D points using Andrew's monotone chain
        algorithm and returns the indices of the hull vertices in counter‑clockwise order.
        """
        # Pair points with their original indices
        idx_pts = sorted(enumerate(points), key=lambda x: (x[1][0], x[1][1]))

        def cross(o: Tuple[int, float, float], a: Tuple[int, float, float], b: Tuple[int, float, float]) -> float:
            return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])

        lower: List[Tuple[int, float, float]] = []
        for idx, (x, y) in idx_pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], (idx, x, y)) <= 0:
                lower.pop()
            lower.append((idx, x, y))

        upper: List[Tuple[int, float, float]] = []
        for idx, (x, y) in reversed(idx_pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], (idx, x, y)) <= 0:
                upper.pop()
            upper.append((idx, x, y))

        # Concatenate lower and upper hulls, removing the last point of each
        # because it is repeated at the beginning of the other list.
        full = lower[:-1] + upper[:-1]
        return [p[0] for p in full]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the Convex Hull problem without using SciPy.

        :param problem: A dictionary with key 'points' mapping to a list of coordinate pairs.
        :return: A dictionary with keys:
                 "hull_vertices": List of indices of the points that form the convex hull.
                 "hull_points": List of coordinates of the points that form the convex hull.
        """
        points = problem["points"]
        # Ensure list of tuples for numeric operations
        pts = [(float(x), float(y)) for x, y in points]
        hull_indices = self._convex_hull(pts)
        hull_points = [points[i] for i in hull_indices]
        return {"hull_vertices": hull_indices, "hull_points": hull_points}