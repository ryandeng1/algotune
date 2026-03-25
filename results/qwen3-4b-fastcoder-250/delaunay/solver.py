import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull
        
        sorted_simplices = np.sort(simplices, axis=1)
        sorted_convex_hull = np.sort(convex_hull, axis=1)
        
        simplices_list = [tuple(row) for row in sorted_simplices]
        convex_hull_list = [tuple(edge) for edge in sorted_convex_hull]
        
        return {
            "simplices": simplices_list,
            "convex_hull": convex_hull_list
        }
