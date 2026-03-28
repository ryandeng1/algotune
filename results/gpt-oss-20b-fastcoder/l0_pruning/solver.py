from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Return the vector consisting of the k largest‑in‑absolute value terms
        from the input vector v; all other terms are set to 0.
        """
        v = np.asarray(problem.get("v", []), dtype=float).ravel()
        k = int(problem.get("k", 0))
        if k <= 0 or v.size == 0:
            return {"solution": [0.0] * v.size}

        # Find the indices of the k largest absolute values in linear time
        idx = np.argpartition(np.abs(v), -k)[-k:]
        # Create the pruned vector
        pruned = np.empty_like(v)
        pruned.fill(0.0)
        pruned[idx] = v[idx]
        return {"solution": pruned.tolist()}