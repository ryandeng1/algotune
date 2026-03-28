from typing import Any
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = problem["points"]
        vor = ScipyVoronoi(points)
        
        vertices = vor.vertices.tolist()
        regions = vor.regions
        point_region = np.arange(len(points))
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = vor.ridge_vertices.tolist()
        
        regions = [regions[i] for i in vor.point_region]
        
        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }