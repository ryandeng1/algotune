from typing import Any, List, Tuple

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Compute the 2‑D convex hull of a set of points using the
        monotone chain algorithm (O(n log n)).  This implementation
        avoids importing heavy libraries such as scipy and is
        therefore much faster for problems that fit into memory.

        Parameters
        ----------
        problem : dict
            Expected to contain a key ``"points"`` with a list of
            ``[x, y]`` coordinates.

        Returns
        -------
        dict
            ``"hull_vertices"`` – list of indices of the input points
            that form the convex hull (in counter‑clockwise order).
            ``"hull_points"`` – the corresponding coordinate pairs.
        """
        points: List[List[float]] = problem["points"]
        if len(points) < 3:
            return {"hull_vertices": list(range(len(points))), "hull_points": points}

        # Attach original indices so we can return them later
        indexed = [(x, y, i) for i, (x, y) in enumerate(points)]

        # Sort by (x, y)
        indexed.sort(key=lambda p: (p[0], p[1]))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in indexed:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(indexed):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate lower and upper to get full hull, removing duplicate endpoints
        hull = lower[:-1] + upper[:-1]

        hull_vertices = [p[2] for p in hull]
        hull_points = [[p[0], p[1]] for p in hull]

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}