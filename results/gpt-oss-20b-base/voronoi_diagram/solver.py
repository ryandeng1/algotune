# solver.py

import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute the Voronoi diagram of the given set of 2D points.
        The output follows the format required by the verification harness.
        """
        # Extract points as a NumPy array
        points = np.asarray(problem["points"])

        # Compute the Voronoi diagram
        vor = ScipyVoronoi(points)

        # Build the solution dictionary
        solution = {
            "vertices": vor.vertices.tolist(),
            "regions": [list(region) for region in vor.regions],
            "point_region": np.arange(len(points)),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices.tolist(),
        }

        # Reorder regions according to point_region mapping
        solution["regions"] = [solution["regions"][idx] for idx in vor.point_region]

        return solution
