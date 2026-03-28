import numpy as np
from scipy.spatial import Delaunay as SciPyDelaunay

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        pts = np.asarray(problem['points'])
        tri = SciPyDelaunay(pts)
        simp = tri.simplices
        hull = tri.convex_hull

        # canonicalize simplices: sort vertex indices inside each simplex
        # and then sort the list of simplices
        simp_sorted = np.sort(simp, axis=1)
        simp_tuples = [tuple(row) for row in simp_sorted]
        simp_tuples.sort()

        # canonicalize hull edges: undirected edges -> sorted vertex indices per edge
        hull_sorted = np.sort(hull, axis=1)
        hull_tuples = [tuple(row) for row in hull_sorted]
        hull_tuples.sort()

        return {'simplices': simp_tuples, 'convex_hull': hull_tuples}