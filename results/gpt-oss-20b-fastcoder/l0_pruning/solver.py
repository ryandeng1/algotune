from typing import Any, Dict, List
import numpy as np

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, List]:
        """
        Solve the l0 pruning problem: keep only the k elements of v with
        the largest absolute values, setting all others to zero.
        """
        v = np.asarray(problem['v'])
        k = int(problem['k'])

        # Work with a 1-D view of the data
        v_flat = v.ravel()

        # Find indices of the k largest-magnitude elements in linear time
        if k > 0:
            # argpartition returns indices that would partition the array
            idx_top = np.argpartition(-np.abs(v_flat), k - 1)[:k]
        else:
            idx_top = np.array([], dtype=int)

        # Reconstruct the full solution array
        solution_arr = np.zeros_like(v_flat)
        solution_arr[idx_top] = v_flat[idx_top]

        # Return as a Python list
        return {'solution': solution_arr.tolist()}