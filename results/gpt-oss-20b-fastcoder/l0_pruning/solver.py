import numpy as np
from typing import Any, Dict, List

class Solver:

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the problem using the algorithm described in
        https://doi.org/10.1109/CVPR.2018.00890.

        The task is to keep the k largest-magnitude elements of v and set the rest to 0.
        The implementation uses np.argpartition which runs in O(n) average time.
        """
        # Extract and flatten the input array
        v = np.asarray(problem['v']).ravel()

        # Number of elements to keep
        k = int(problem['k'])
        n = v.size

        if k <= 0:
            # Nothing to keep – return all zeros
            pruned = np.zeros(n, dtype=v.dtype)
        elif k >= n:
            # Keep everything
            pruned = v.copy()
        else:
            # Find indices of the k elements with largest absolute value
            # np.argpartition returns indices such that the elements at those indices
            # are the k largest (in any order)
            idx = np.argpartition(-np.abs(v), k - 1)[:k]

            # Build the result array
            pruned = np.zeros(n, dtype=v.dtype)
            pruned[idx] = v[idx]

        # Return as plain Python list
        return {'solution': pruned.tolist()}