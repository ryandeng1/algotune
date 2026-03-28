import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Retain the k largest magnitude entries of the vector v and zero out the rest.
        """
        v = np.asarray(problem['v'])
        k = int(problem['k'])

        # Flatten to 1‑D array
        v = v.ravel()

        # Find the indices of the k largest magnitude values
        if k >= v.size:
            # All elements are kept
            pruned = v.copy()
        else:
            # Partition once to get the k largest magnitudes
            idx = np.argpartition(np.abs(v), -k)[-k:]
            mask = np.zeros(v.shape, dtype=bool)
            mask[idx] = True
            pruned = np.where(mask, v, 0.0)

        return {"solution": pruned.tolist()}