from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the problem using the algorithm described in https://doi.org/10.1109/CVPR.2018.00890.
        This optimization problem has quadratic objective and non-convex constraints.
        However, it can be solved exactly in O(nlogn) time via stable sorting algorithm.

        :param problem: A dictionary of the problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the l0_pruning task.
        """
        # Get data and flatten in a single view
        v = np.asarray(problem.get('v')).ravel()
        k = int(problem.get('k'))

        # Number of elements must be at least k
        n = v.size
        if k >= n:
            # All elements are kept
            return {"solution": v.tolist()}

        # Select indices of the k largest absolute values
        abs_v = np.abs(v)
        topk_idx = np.argpartition(abs_v, -k)[-k:]

        # Build the pruned vector
        pruned = np.zeros_like(v)
        pruned[topk_idx] = v[topk_idx]

        return {"solution": pruned.tolist()}