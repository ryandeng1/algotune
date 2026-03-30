from typing import Any
import numpy as np

class Solver:
    """
    Solver for the l0‐pruning problem.
    The algorithm keeps the k largest absolute values of the input vector `v`
    and zeroes everything else.  Using `np.argpartition` gives an
    average linear time solution (O(n)).
    """

    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        v = np.asarray(problem.get('v'))
        k = problem.get('k')

        # Flatten v (handled by np.asarray) and keep the shape
        shape = v.shape
        v_flat = v.ravel()

        # Find indices of the k largest absolute values.
        # We want the largest, so we partition with the negative values.
        abs_v = np.abs(v_flat)
        if k >= abs_v.size:
            # All elements are kept
            pruned_flat = v_flat.copy()
        else:
            # Use argpartition for linear time selection
            partition_idx = np.argpartition(-abs_v, k - 1)[:k]
            # Initialize zeros
            pruned_flat = np.zeros_like(v_flat)
            # Place the chosen values
            pruned_flat[partition_idx] = v_flat[partition_idx]

        # Reshape back to the original shape and convert to list
        pruned = pruned_flat.reshape(shape).tolist()
        return {"solution": pruned}