import numpy as np
from typing import Any, Dict, List

class Solver:
    @staticmethod
    def _monotone_chain(points: np.ndarray, indices: np.ndarray) -> List[int]:
        # Sort by x, then y
        order = np.lexsort((points[:, 1], points[:, 0]))
        pts = points[order]
        idx = indices[order]

        # Build lower hull
        lower = []
        for p, i in zip(pts, idx):
            while len(lower) >= 2:
                p1, p2 = lower[-2][0], lower[-1][0]
                if (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) <= 0:
                    lower.pop()
                else:
                    break
            lower.append((p, i))

        # Build upper hull
        upper = []
        for p, i in zip(reversed(pts), reversed(idx)):
            while len(upper) >= 2:
                p1, p2 = upper[-2][0], upper[-1][0]
                if (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]) <= 0:
                    upper.pop()
                else:
                    break
            upper.append((p, i))

        # Concatenate lower and upper removing duplicate endpoints
        hull = lower[:-1] + upper[:-1]
        return [i for _, i in hull]

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        pts = np.asarray(problem["points"], dtype=np.float64)
        n = pts.shape[0]
        if n < 3:
            # For trivial cases return all points as hull
            hull_vertices = list(range(n))
            hull_points = pts.tolist()
            return {"hull_vertices": hull_vertices, "hull_points": hull_points}

        indices = np.arange(n)
        hull_indices = self._monotone_chain(pts, indices)
        hull_points = pts[hull_indices].tolist()
        return {"hull_vertices": hull_indices.tolist(), "hull_points": hull_points}
