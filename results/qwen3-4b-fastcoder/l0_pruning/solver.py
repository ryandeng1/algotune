from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.array(problem.get("v"))
        k = problem.get("k")

        v = v.reshape(-1)
        indx = np.argsort(np.abs(v), kind='stable')
        remaining_indx = indx[-k:]
        pruned = np.zeros_like(v)
        pruned[remaining_indx] = v[remaining_indx]

        return {"solution": pruned.tolist()}