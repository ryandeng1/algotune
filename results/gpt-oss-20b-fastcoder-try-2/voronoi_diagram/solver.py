import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a Voronoi diagram for the supplied points using scipy's fast implementation
        and return the result in a lightweight dictionary format.

        Parameters
        ----------
        problem : dict
            Dictionary with a single key ``"points"`` containing an N x D array of points.

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - "vertices": list of Voronoi vertex coordinates.
            - "regions" : list of lists, each sub-list holds the indices of vertices
                          that form the closed polygon of a Voronoi cell
                          (indices are into the ``"vertices"`` list).
            - "point_region" : 1‑D array mapping each input point to the index of its cell
                               in the ``"regions"`` list.
            - "ridge_points" : list of point index pairs that share a Voronoi ridge.
            - "ridge_vertices": list of vertex index pairs that form each ridge.
        """
        # Grab points, enforcing a numpy array for fast processing
        pts = np.asarray(problem["points"], dtype=np.float64, order="C")

        # Compute the Voronoi diagram with SciPy (C‑extension, fast)
        vor = ScipyVoronoi(pts)

        # Convert to lightweight Python structures only once
        vertices = [tuple(v) for v in vor.vertices]          # list of tuples
        # Map each point to its corresponding Voronoi region
        point_region = vor.point_region.astype(int).tolist()
        # Build the list of vertex indices that form each region (for finite ones only)
        regions = [list(r) for r in vor.regions]
        # re‑order regions so that regions[i] corresponds to point i
        regions = [regions[pr] for pr in point_region]
        ridge_points = [tuple(p) for p in vor.ridge_points]
        ridge_vertices = [tuple(v) for v in vor.ridge_vertices]

        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }