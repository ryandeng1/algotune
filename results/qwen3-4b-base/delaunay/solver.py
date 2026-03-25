import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem["points"])
        tri = SciPyDelaunay(pts)
        simplices = tri.simplices
        convex_hull = tri.convex_hull

        simplices = np.sort(simplices, axis=1)
        simplices = simplices[np.argsort(simplices)]

        convex_hull = np.sort(convex_hull, axis=1)
        convex_hull = convex_hull[np.argsort(convex_hull)]

        return {
            "simplices": [tuple(s) for s in simplices],
            "convex_hull": [tuple(e) for e in convex_hull]
        }
