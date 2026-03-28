from typing import Any
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        # Extract points as a NumPy array for speed
        points = np.asarray(problem["points"])
        # Construct the Voronoi diagram using SciPy
        vor = ScipyVoronoi(points)

        # Build the solution dictionary with minimal processing
        return {
            "vertices": vor.vertices.tolist(),
            "regions": [vor.regions[i] for i in vor.point_region],
            "point_region": vor.point_region.tolist(),
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices.tolist(),
        }