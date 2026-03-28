from typing import Any, Dict, List, Tuple

class Solver:
    """
    A fast pure‑Python implementation of the 2‑D convex hull.
    Uses the monotone‑chain (Andrew) algorithm: O(n log n) time
    and O(n) additional memory.  All point indices are preserved.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        points: List[List[float]] = problem["points"]
        n = len(points)

        # Attach original indices to points
        pts_with_idx: List[Tuple[float, float, int]] = [
            (x, y, i) for i, (x, y) in enumerate(points)
        ]

        # Sort lexicographically: by x, then by y
        pts_with_idx.sort(key=lambda p: (p[0], p[1]))

        def cross(o: Tuple[float, float, int], a: Tuple[float, float, int], b: Tuple[float, float, int]) -> float:
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float, int]] = []
        for p in pts_with_idx:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        upper: List[Tuple[float, float, int]] = []
        for p in reversed(pts_with_idx):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenate lower and upper to get full hull (excluding duplicate endpoints)
        hull = lower[:-1] + upper[:-1]

        hull_vertices = [idx for _, _, idx in hull]
        hull_points = [[points[i][0], points[i][1]] for _, _, i in hull]

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}