from typing import Any, Dict, List
import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct the Voronoi diagram using scipy.spatial.Voronoi and
        return the data in the requested format.
        """
        points = problem["points"]

        # Build the Voronoi diagram – this is the expensive part.
        vor = ScipyVoronoi(points)

        # Convert the NumPy outputs to plain Python containers once.
        vertices: List[List[float]] = vor.vertices.tolist()
        # Keep the original ordering of regions: each point has its own region
        # given by vor.point_region.  The 'regions' attribute contains all
        # regions (including -1 for open ones) – we therefore just map the
        # appropriate ones.
        point_region = vor.point_region  # already a 1‑D NumPy array
        regions: List[List[int]] = [vor.regions[idx] for idx in point_region]

        ridge_points: List[List[int]] = vor.ridge_points.tolist()
        ridge_vertices: List[List[int]] = vor.ridge_vertices.tolist()

        solution: Dict[str, Any] = {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }
        return solution