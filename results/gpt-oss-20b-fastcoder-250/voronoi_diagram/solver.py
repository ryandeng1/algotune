import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        points = np.asarray(problem["points"])
        vor = ScipyVoronoi(points)

        # Directly build the solution using native Python structures
        vertices = vor.vertices.tolist()
        regions = [list(region) for region in vor.regions]
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = vor.ridge_vertices

        # Map each input point to its Voronoi region
        point_region = list(vor.point_region)

        # Prepare final structure
        solution = {
            "vertices": vertices,
            "regions": [regions[idx] for idx in point_region],
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }
        return solution