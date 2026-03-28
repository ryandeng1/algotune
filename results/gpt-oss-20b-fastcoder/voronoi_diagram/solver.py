import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        points = np.asarray(problem['points'], dtype=np.float64)
        vor = ScipyVoronoi(points)

        # Build the solution dictionary in a compact way
        vertices = vor.vertices.tolist()
        regions = [list(r) for r in vor.regions]
        point_region = vor.point_region.tolist()
        ridge_points = vor.ridge_points.tolist()
        ridge_vertices = vor.ridge_vertices.tolist()

        # Reorder regions according to the point_region mapping
        regions = [regions[i] for i in point_region]

        return {
            'vertices': vertices,
            'regions': regions,
            'point_region': point_region,
            'ridge_points': ridge_points,
            'ridge_vertices': ridge_vertices,
        }