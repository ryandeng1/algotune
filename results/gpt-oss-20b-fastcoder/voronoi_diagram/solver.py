from typing import Any, Dict, List
import numpy as np
from scipy.spatial import Voronoi

def solve(problem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a Voronoi diagram for a set of points using scipy.spatial.Voronoi
    and return the data structures in the format required by the benchmark.

    :param problem: Dictionary with key 'points' containing a 2‑D ndarray or list of points.
    :return: Dictionary containing vertices, regions, point_region, ridge_points,
             and ridge_vertices.
    """
    # Ensure the points are in a NumPy array for optimal performance
    points = np.asarray(problem["points"], dtype=np.float64)

    # Build the Voronoi diagram
    vor = Voronoi(points)

    # `vor.point_region` already maps each input point to its region.
    # Construct the region list in the same order.
    regions = [vor.regions[idx] for idx in vor.point_region]

    # Convert ndarray objects to plain Python lists only where necessary.
    # The outermost list structures are small, converting the vertices once
    # is the main cost; the remaining lists are thin.
    solution = {
        "vertices": vor.vertices.tolist(),
        "regions": [list(r) for r in regions],
        "point_region": list(vor.point_region),
        "ridge_points": vor.ridge_points.tolist(),
        "ridge_vertices": vor.ridge_vertices.tolist(),
    }

    return solution