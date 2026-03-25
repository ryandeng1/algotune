import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        """
        Compute the Voronoi diagram for the given set of points.
        The implementation relies on scipy.spatial.Voronoi which produces
        the same output format expected by the validation routine.
        """
        # Extract points from the problem dictionary
        points = np.asarray(problem["points"], dtype=float)

        # Compute Voronoi diagram
        vor = ScipyVoronoi(points)

        # Build the solution dictionary
        solution = {
            "vertices": vor.vertices.tolist(),
            "regions": [list(r) for r in vor.regions],
            "point_region": vor.point_region.tolist(),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices.tolist(),
        }

        # Filter regions for each input point
        solution["regions"] = [solution["regions"][idx] for idx in vor.point_region]

        return solution
