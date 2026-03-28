from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem.get("v"))
        k = problem.get("k")
        v = v.flatten()

        def subproblem_sol(vn, z):
            n = len(vn)
            if n == 0:
                return np.zeros_like(vn)
            mu = np.sort(vn)[::-1]
            cumsum = np.cumsum(mu)
            denom = np.arange(1, n + 1)
            condition = mu < (cumsum - z) / denom
            j_index = np.argmax(condition) if condition.any() else None
            theta = (cumsum[j_index] - z) / (j_index + 1) if j_index is not None else 0.0
            w = np.maximum(vn - theta, 0)
            return w

        u = np.abs(v)
        b = subproblem_sol(u, k)
        new_v = b * np.sign(v)
        pruned = np.where(new_v != 0, new_v, 0)
        solution = {"solution": pruned.tolist()}
        return solution