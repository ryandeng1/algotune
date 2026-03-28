from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        # Extract and flatten the input array
        y = np.asarray(problem.get("y"), dtype=np.float64).ravel()
        n = y.size

        # Quick linear-time algorithm when the sum of all entries is already 1 and all non‑negative
        if y.sum() == 1.0 and y.min() >= 0.0:
            return {"solution": y.tolist()}

        # Sort y in descending order and compute the cumulative sum
        sorted_y = np.sort(y)[::-1]
        sorted_cumsum = np.cumsum(sorted_y)

        # Compute the threshold index rho
        # We look for the largest j such that sorted_y[j] > (sorted_cumsum[j] - 1) / (j + 1)
        j = np.arange(1, n + 1)
        rhs = (sorted_cumsum - 1) / j
        mask = sorted_y > rhs
        if not mask.any():
            # All coordinates are 0, return the zero vector
            return {"solution": [0.0] * n}
        rho = mask.nonzero()[0][-1]

        # Compute theta and the projection
        theta = (sorted_cumsum[rho] - 1) / (rho + 1)
        x = np.maximum(y - theta, 0)

        return {"solution": x.tolist()}