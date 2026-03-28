from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem.get("v"))
        k = problem.get("k")

        # Ensure v is a column vector
        v = v.flatten()

        def subproblem_sol(vn, z):
            vn = np.asarray(vn)
            mu = np.sort(vn, kind="mergesort")[::-1]
            n = len(mu)
            if n == 0:
                return np.zeros_like(vn)
            cumsum = np.cumsum(mu)
            denom = np.arange(1, n + 1)
            rhs = (cumsum - z) / denom
            mask = mu < rhs
            idx = np.where(mask)[0]
            theta = rhs[idx[0]] if idx.size > 0 else 0
            return np.maximum(vn - theta, 0)

        u = np.abs(v)
        b = subproblem_sol(u, k)
        new_v = b * np.sign(v)
        remaining_indx = new_v != 0
        pruned = np.zeros_like(v)
        pruned[remaining_indx] = new_v[remaining_indx]

        solution = {"solution": pruned.tolist()}
        return solution