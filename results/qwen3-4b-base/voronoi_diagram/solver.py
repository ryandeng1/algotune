import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = problem["points"]
        vor = ScipyVoronoi(points)
        n_points = len(points)
        
        vertices = vor.vertices.tolist()
        regions = vor.regions
        point_region = list(range(n_points))
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = [list(v) for v in vor.ridge_vertices]
        
        regions_reordered = [regions[i] for i in vor.point_region]
        
        return {
            "vertices": vertices,
            "regions": regions_reordered,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices
        }
