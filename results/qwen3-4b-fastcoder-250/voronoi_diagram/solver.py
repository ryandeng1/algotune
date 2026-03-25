import numpy as np
from scipy.spatial import Voronoi

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        points = np.array(problem["points"])
        vor = Voronoi(points)
        
        vertices = vor.vertices.tolist()
        regions = vor.regions
        point_region = vor.point_region.tolist()
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = vor.ridge_vertices.tolist()
        
        regions_list = [regions[i] for i in point_region]
        
        return {
            "vertices": vertices,
            "regions": regions_list,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices
        }
