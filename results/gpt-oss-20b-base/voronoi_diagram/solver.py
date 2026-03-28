import numpy as np
from scipy.spatial import Voronoi as ScipyVoronoi

class Solver:
    """
    Fast Voronoi diagram construction using scipy.spatial.Voronoi.
    """

    def solve(self, problem: dict[str, any]) -> dict[str, any]:
        # Extract points once – no extra copies
        points = np.asarray(problem["points"])

        # Compute the Voronoi diagram
        vor = ScipyVoronoi(points)

        # Prepare the output using low‑overhead conversions
        # Vertices: keep as a (n_vertices, 2) float array
        vertices = np.asarray(vor.vertices, dtype=np.float64)

        # Point-region mapping – map each input point to the index of its region
        point_region = np.arange(len(points), dtype=np.int32)

        # Ridge information – keep as integer arrays
        ridge_points = np.asarray(vor.ridge_points, dtype=np.int32)
        ridge_vertices = np.asarray(vor.ridge_vertices, dtype=np.int32)

        # Each point’s region consists of a subset of vertex indices.
        # We avoid converting all regions; just slice the relevant ones.
        regions = [vor.regions[idx] for idx in point_region]

        # Build the solution dictionary.
        # No intermediate list conversions – just store the NumPy arrays.
        return {
            "vertices": vertices,
            "regions": regions,
            "point_region": point_region,
            "ridge_points": ridge_points,
            "ridge_vertices": ridge_vertices,
        }