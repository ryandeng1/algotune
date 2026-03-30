from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi


class Solver:
    """
    Thin wrapper around ``scipy.spatial.Voronoi`` that returns
    the result in the exact dictionary format required by the
    tests.  The implementation is already O(n log n) in the number
    of points and the only non‑trivial part is the construction of
    the `regions` list that excludes the ``-1`` indices used by
    SciPy to denote vertices at infinity.
    """

    @staticmethod
    def _filter_region(region: List[int]) -> List[int]:
        """Remove the ``-1`` markers from a region."""
        return [idx for idx in region if idx != -1]

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        points = np.asarray(problem["points"], dtype=np.float64)
        vor = ScipyVoronoi(points)

        # Convert the vertices and ridge points to plain Python lists
        vertices: List[List[float]] = vor.vertices.tolist()
        ridge_points: List[List[int]] = vor.ridge_points.tolist()
        ridge_vertices: List[List[int]] = [list(rv) for rv in vor.ridge_vertices]

        # Build the per‑point region list, filtering out ``-1`` entries
        point_region = list(vor.point_region)  # 0‑based indices matching points
        regions: List[List[int]] = []
        for idx in point_region:
            regions.append(self._filter_region(vor.regions[idx]))

        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }