import numpy as np
from typing import Any, Dict, List


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the l0 pruning problem in O(n) expected time by selecting the
        k elements with the largest absolute value. This implementation uses
        NumPy's efficient argpartition instead of a full stable sort.

        Args:
            problem: Dictionary containing the inputs:
                - "v": Sequence of numbers (can be nested or array-like).
                - "k": Integer, number of elements to keep.

        Returns:
            Dictionary with key `"solution"` containing a list of length n, where
            all but the k largest-in-absolute-value elements are zeroed.
        """
        v = np.asarray(problem["v"]).flatten()
        k = int(problem["k"])

        # Guard against invalid k
        if k < 0:
            raise ValueError("k must be non‑negative")
        n = v.size
        if k > n:
            k = n

        # If no elements are to be kept, return all zeros
        if k == 0:
            return {"solution": [0.0] * n}

        # Find indices of the k largest absolute values
        # argpartition gives the indices of the k largest in any order
        topk_idx = np.argpartition(np.abs(v), -k)[-k:]

        # Build the pruned array
        pruned = np.zeros_like(v, dtype=float)
        pruned[topk_idx] = v[topk_idx]

        return {"solution": pruned.tolist()}