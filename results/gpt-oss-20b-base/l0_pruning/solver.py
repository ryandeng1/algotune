from typing import Any, Dict, List
import numpy as np


class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the problem using the algorithm described in
        https://doi.org/10.1109/CVPR.2018.00890.
        This optimization problem has quadratic objective and non‑convex constraints.
        However, it can be solved exactly in O(n log n) time via stable sorting algorithm.

        :param problem: A dictionary of the problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the l0_pruning task.
        """
        # Pull input parameters
        v = np.asarray(problem["v"], dtype=float).flatten()
        k = int(problem["k"])

        # Guard against k larger than length of v
        n = v.size
        if k >= n:
            return {"solution": v.tolist()}

        # Find the indices of the k largest absolute values using a partial
        # selection algorithm (argpartition) – O(n) average time.
        # We then sort those k values stably in descending order of |v|.
        # This mirrors the behaviour of the original mergesort approach.
        top_k_idx = np.argpartition(-np.abs(v), k - 1)[:k]
        # Stable sort the selected indices by absolute value
        top_k_idx = top_k_idx[np.argsort(np.abs(v[top_k_idx]), kind="mergesort")]

        # Build the pruned vector (zero everywhere except the chosen indices)
        pruned = np.zeros_like(v)
        pruned[top_k_idx] = v[top_k_idx]

        return {"solution": pruned.tolist()}