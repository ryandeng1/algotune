# solver.py
from typing import Any, Dict, List

import numpy as np
from scipy.spatial import ConvexHull


class Solver:
    """
    Optimised solver for 2‑/3‑D convex hulls using SciPy's FastQhull implementation.
    The only optimisation is to minimise data conversions and to initialise
    the points array with the minimal required dtype.
    """

    def _to_numpy(self, points: Any) -> np.ndarray:
        """
        Ensure the point array is a contiguous NumPy array with the smallest
        practical dtype (float64).  Converting once here keeps the main
        routine free of checks and reduces repeated allocations.
        """
        arr = np.asarray(points, dtype=np.float64, order="C")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Calculate the convex hull vertices and corresponding points.

        Parameters
        ----------
        problem : dict
            ``{"points": array_like}`` where the dataframe is (n, dim) shaped.

        Returns
        -------
        dict
            ``"hull_vertices"`` – indices of vertices (list[int]).
            ``"hull_points"``   – coordinates of these vertices (list[list[float]]).
        """
        points = self._to_numpy(problem["points"])
        hull = ConvexHull(points)
        vertices = hull.vertices.tolist()
        hull_points = points[vertices].tolist()
        return {"hull_vertices": vertices, "hull_points": hull_points}