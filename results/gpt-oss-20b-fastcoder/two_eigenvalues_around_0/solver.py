from typing import Any
import numpy as np
from heapq import nsmallest

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Efficiently find the two eigenvalues closest to zero of a symmetric matrix.

        Uses numpy to compute all eigenvalues once and then retrieves the two with
        the smallest absolute value without fully sorting the array.

        Args:
            problem: Dict with key 'matrix' containing a list of lists of floats.

        Returns:
            List of the two eigenvalues closest to zero, sorted by increasing absolute value.
        """
        # Convert to a NumPy array once (the input can be a list of lists)
        matrix = np.array(problem['matrix'], dtype=float)

        # Compute all eigenvalues (for symmetric matrices use the efficient routine)
        eigs = np.linalg.eigvalsh(matrix)

        # Retrieve the two eigenvalues with the smallest absolute value
        # without fully sorting:  use nsmallest from heapq
        two_smallest = nsmallest(2, eigs, key=abs)

        # Ensure they are returned sorted by absolute value
        two_smallest.sort(key=abs)
        return two_smallest