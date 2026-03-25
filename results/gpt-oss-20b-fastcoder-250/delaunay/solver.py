import numpy as np
from scipy.spatial import Delaunay

class Solver:
    def solve(self, problem: dict) -> dict:
        pts = np.asarray(problem["points"])
        tri = Delaunay(pts)
        return {
            "simplices": tri.simplices,
            "convex_hull": tri.convex_hull
        }
