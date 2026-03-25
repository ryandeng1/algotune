import numpy as np
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the convex hull of a set of 2D points using the monotone chain algorithm.

        Parameters
        ----------
        problem : dict
            Must contain the key "points" with an array-like of shape (n, 2).

        Returns
        -------
        dict
            {"hull_vertices": [...], "hull_points": [...]}
            where hull_vertices are indices into the original point array in
            counter‑clockwise order and hull_points are the corresponding coordinates.
        """
        points = np.asarray(problem["points"], dtype=float)
        if points.shape[0] == 0:
            return {"hull_vertices": [], "hull_points": []}

        n = points.shape[0]
        # Keep original indices
        idx = np.arange(n)

        # Sort points by x, then y
        order = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[order]
        id_order = idx[order]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        lower_idx = []
        for p, ii in zip(pts, id_order):
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
                lower_idx.pop()
            lower.append(p)
            lower_idx.append(ii)

        # Build upper hull
        upper = []
        upper_idx = []
        for p, ii in zip(reversed(pts), reversed(id_order)):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
                upper_idx.pop()
            upper.append(p)
            upper_idx.append(ii)

        # Concatenate, removing duplicates at the endpoints
        hull_pts = np.vstack((lower, upper[1:-1]))
        hull_idx = np.hstack((lower_idx, upper_idx[1:-1]))

        # Ensure at least 3 vertices (for degenerate cases return all unique)
        if hull_idx.shape[0] < 3:
            uniq = np.unique(order)
            hull_idx = order[uniq]
            hull_pts = points[hull_idx]

        # Convert to lists for output
        hull_vertices = hull_idx.tolist()
        hull_points = hull_pts.tolist()

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}
