from typing import Any
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Build a Voronoi diagram with scipy and return a compact representation.

        :param problem: Dictionary containing at least the key 'points' with an
                        N x 2 array-like of coordinates.
        :return: Dictionary with the diagram data.
        """
        points = np.asarray(problem["points"], dtype=np.float64, order="C")
        vor = ScipyVoronoi(points)

        # Vertices and ridge vertices: already numpy arrays – convert once to list.
        vertices = vor.vertices.tolist()
        ridge_vertices = vor.ridge_vertices.tolist()
        ridge_points = vor.ridge_points.tolist()

        # Construct region list in the same order as the input points.
        regions = [vor.regions[i] for i in vor.point_region]
        # Convert each region from tuple to list to avoid external mutation.
        regions = [list(r) for r in regions]

        # Map each point to its region index.
        point_region = np.arange(len(points), dtype=np.int64)

        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }