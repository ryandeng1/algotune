from typing import List
import numpy as np


class Solver:
    def solve(self, problem: dict[str, List[List[float]]]) -> List[float]:
        """
        Return the two eigenvalues of the given symmetric matrix that are closest
        to zero (by absolute value). The result is sorted from the smallest
        absolute value to the larger one.
        """
        # Convert the input into a numpy array of type float64.
        mat = np.asarray(problem["matrix"], dtype=np.float64)

        # Compute all eigenvalues using an efficient routine for symmetric
        # matrices. This is O(n³) but the fastest available method.
        vals = np.linalg.eigvalsh(mat)

        # Find indices of the two values with smallest absolute value.
        # np.argpartition gives an O(n) selection.
        idx = np.argpartition(np.abs(vals), 2)[:2]

        # Sort the selected eigenvalues by absolute value.
        selected = vals[idx]
        # Sorting by absolute value while retaining the actual eigenvalues.
        return sorted(selected, key=abs)