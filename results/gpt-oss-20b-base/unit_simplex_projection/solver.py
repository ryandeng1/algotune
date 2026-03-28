from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        y = problem["y"]
        # Ensure a 1‑D NumPy array for vectorised operations
        y_arr = (y if isinstance(y, (list, tuple)) else y).copy()
        if not hasattr(y_arr, "__len__"):
            y_arr = [y_arr]

        y_arr = y_arr if isinstance(y_arr, list) else y_arr.tolist()
        n = len(y_arr)

        # Fast projection onto the probability simplex
        # 1. Sort in descending order
        # 2. Find the largest k such that y_sorted[k] > (sum(y_sorted[:k+1]) - 1) / (k+1)
        # 3. Compute theta and project
        import numpy as np
        y_np = np.array(y_arr, dtype=float)
        # descending sort
        idx = np.argsort(-y_np)
        y_sorted = y_np[idx]
        # cumulative sum offset by -1
        cumsum = np.cumsum(y_sorted) - 1.0
        # create divisor array
        div = np.arange(1, n + 1, dtype=float)
        # boolean mask where condition holds
        mask = y_sorted > cumsum / div
        # last true index
        rho = mask.argmax() if not mask.all() else n - 1
        # compute theta
        theta = cumsum[rho] / (rho + 1.0)
        # projection
        proj = y_np - theta
        proj[proj < 0] = 0.0
        return {"solution": proj.tolist()}