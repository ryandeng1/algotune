from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem.get("v"))
        k = problem.get("k")

        v = v.ravel()

        pruned = np.zeros_like(v)
        indx = np.argsort(np.abs(v), kind="mergesort")
        remaining_indx = indx[-k:]
        pruned[remaining_indx] = v[remaining_indx]

        return {"solution": pruned.tolist()}