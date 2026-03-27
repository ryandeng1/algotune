from typing import Any, Dict, List
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the l1-pruning problem in O(n log n).
        """
        v = np.asarray(problem["v"], dtype=float).ravel()
        k = float(problem["k"])

        # Subproblem solver using vectorised operations
        def subproblem_sol(vn: np.ndarray, z: float) -> np.ndarray:
            # Sorted descending
            mu = np.sort(vn)[::-1]
            cs = np.cumsum(mu)
            idx = np.arange(len(mu))
            rhs = (cs - z) / (idx + 1)
            # Find first index where mu < rhs
            mask = mu < rhs
            if not np.any(mask):
                theta = 0.0
            else:
                j = mask.argmax()
                theta = rhs[j]
            w = np.maximum(vn - theta, 0.0)
            return w

        u = np.abs(v)
        b = subproblem_sol(u, k)
        new_v = b * np.sign(v)

        solution = {"solution": new_v.tolist()}
        return solution