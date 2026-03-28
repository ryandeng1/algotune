from typing import Any
import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.array(problem["points"], dtype=np.float64)
        
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        
        canonical_simplices = np.sort(simplices, axis=1)
        n = len(convex_hull)
        canonical_edges = []
        if n > 0:
            canonical_edges = np.column_stack([
                convex_hull,
                np.roll(convex_hull, -1)
            ])
        
        return {
            "simplices": canonical_simplices.tolist(),
            "convex_hull": canonical_edges.tolist()
        }