import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Returns the projection of v onto the L0 ball of radius k.
        Equivalent to keeping the k largest-magnitude elements of v and zeroing the rest.
        """
        v = np.asarray(problem.get("v", []), dtype=np.float64)
        k = int(problem.get("k", 0))

        if v.ndim == 0:          # scalar case
            v = v.reshape(1)
        else:
            v = v.reshape(-1)    # 1‑D view

        n = v.size
        if k <= 0:
            return {"solution": [0.0] * n}

        if k >= n:
            # all entries kept
            return {"solution": v.tolist()}

        # Find indices of the k largest absolute values
        # argpartition gives unsorted result; we don't need order
        largest_idx = np.argpartition(np.abs(v), -k)[-k:]
        result = np.zeros_like(v)
        result[largest_idx] = v[largest_idx]
        return {"solution": result.tolist()}
