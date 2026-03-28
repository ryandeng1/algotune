from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the problem in O(n) expected time using np.argpartition.
        """
        v = np.asarray(problem.get('v')).ravel()
        k = int(problem.get('k'))
        n = v.size
        # choose indices of the k largest absolute values
        top_k = np.argpartition(-np.abs(v), k - 1)[:k]
        # build pruned array
        pruned = np.zeros_like(v)
        pruned[top_k] = v[top_k]
        return {'solution': pruned.tolist()}