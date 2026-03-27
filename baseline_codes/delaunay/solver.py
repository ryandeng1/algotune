from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])

        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        result = {
            "simplices": self._canonical_simplices(simplices),
            "convex_hull": self._canonical_edges(convex_hull),
        }
        return result
