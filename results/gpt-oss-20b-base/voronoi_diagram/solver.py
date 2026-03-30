import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any, Dict, List

class Solver:
    """
    Optimised Voronoi solver for Python 3.10.
    """
    def __init__(self) -> None:
        # No heavy pre‑computations are needed; the cost is dominated by
        # the construction of the Voronoi diagram.
        pass

    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the Voronoi diagram for the given set of points.

        Parameters
        ----------
        problem : dict
            Must contain the key 'points' mapping to an (N, D) array‑like
            of coordinates.

        Returns
        -------
        dict
            Dictionary containing
                - 'vertices'      : list of vertex coordinates
                - 'regions'       : list of regions per input point
                - 'point_region'  : list mapping each point to its region
                - 'ridge_points'  : list of point‑pairs sharing a ridge
                - 'ridge_vertices': list of vertex‑index pairs for each ridge
        """
        # Extract points and build the diagram
        points = np.asarray(problem["points"], dtype=np.float64, order="C")
        vor = ScipyVoronoi(points)

        # Mapping from point index to its region index in vor.regions
        point_region = vor.point_region.tolist()

        # Build the region list for each point
        regions: List[List[int]] = [list(vor.regions[idx]) for idx in point_region]

        # Assemble the solution dictionary
        solution = {
            "vertices": vor.vertices.tolist(),
            "regions": regions,
            "point_region": point_region,
            "ridge_points": vor.ridge_points.tolist(),
            "ridge_vertices": vor.ridge_vertices.tolist(),  # converting to list for consistency
        }
        return solution