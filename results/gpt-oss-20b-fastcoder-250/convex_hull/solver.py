import numpy as np
from scipy.spatial import ConvexHull
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute the convex hull of a set of points.

        Parameters
        ----------
        problem : dict
            Must contain key 'points', a 2‑D array‑like of shape (n, d)
            with the coordinates of the points.

        Returns
        -------
        dict
            ``{'hull_vertices': List[int], 'hull_points': np.ndarray}``
        """
        # Ensure the points are in a NumPy array of the right shape
        pts = np.asarray(problem["points"], dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("Input points must be a 2-D array")

        # Compute the convex hull once with SciPy
        hull = ConvexHull(pts)

        # Convert vertex indices to a plain Python list (1‑based if needed)
        hull_vertices = hull.vertices.tolist()

        # Return the hull points as an array for efficiency
        hull_points = pts[hull.vertices, :]

        return {"hull_vertices": hull_vertices, "hull_points": hull_points}