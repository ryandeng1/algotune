from typing import Any
import numpy as np

class Solver:

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the problem using the algorithm described in
        https://doi.org/10.1109/CVPR.2018.00890.

        This optimization problem is a Quadratic Program (QP).
        However, it can be solved exactly in O(n log n).

        :param problem: A dictionary of the problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution
                 to the l1_pruning task.
        """
        v = np.array(problem['v'], dtype=np.float64)
        k = float(problem['k'])
        v = v.ravel()

        # sub‑problem solver (vectorised)
        def subproblem_sol(vn: np.ndarray, z: float) -> np.ndarray:
            # sort descending
            mu = np.sort(vn)[::-1]
            # cumulative sum of sorted values
            cum = np.cumsum(mu, dtype=np.float64)
            # indices 1..n
            rn = np.arange(1, mu.size + 1, dtype=np.float64)
            # candidate thresholds
            theta_candidate = (cum - z) / rn
            # condition: mu[j] < theta_candidate[j]
            cond = mu < theta_candidate
            # first index where condition holds
            j = np.argmax(cond)
            if cond[j]:
                theta = theta_candidate[j]
            else:
                theta = 0.0
            # compute solution
            return np.maximum(vn - theta, 0.0)

        u = np.abs(v)
        b = subproblem_sol(u, k)
        new_v = b * np.sign(v)

        # Prune zeros
        pruned = np.where(new_v != 0, new_v, 0.0)
        return {'solution': pruned.tolist()}