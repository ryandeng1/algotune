from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull

        canonical_simplices = np.sort(simplices, axis=1)

        n = len(convex_hull)
        if n == 0:
            edges = []
        else:
            edges = np.column_stack((convex_hull[:-1], convex_hull[1:]))
            edges = np.vstack((edges, np.array([convex_hull[-1], convex_hull[0]])))
        
        return {
            "simplices": canonical_simplices.tolist(),
            "convex_hull": edges.tolist()
        }