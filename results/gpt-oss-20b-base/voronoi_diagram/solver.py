from typing import Any, Dict, List
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a Voronoi diagram using SciPy and return it in the required
        dictionary format, with minimal temporary Python objects.
        """
        points = problem["points"]
        points_arr = np.asarray(points, dtype=float)

        # Compute the Voronoi diagram
        vor = ScipyVoronoi(points_arr)

        # Convert the result to the expected format
        solution: Dict[str, Any] = {
            "vertices": vor.vertices.tolist(),
            # Convert each region from tuple to list
            "regions": [list(region) for region in vor.regions],
            # Mapping from input point to the Voronoi region that surrounds it
            "point_region": np.arange(len(points_arr)).tolist(),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices.tolist(),
        }

        # Reorder regions according to `point_region`
        solution["regions"] = [solution["regions"][idx] for idx in vor.point_region]

        return solution