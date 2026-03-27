import numpy as np
from typing import List


class Solver:
    def solve(self, problem: List[float]) -> List[complex]:
        """
        Find all roots of the polynomial with real coefficients.

        Parameters
        ----------
        problem:
            Coefficients [a_n, a_{n-1}, ..., a_0] in descending order.

        Returns
        -------
        List[complex]
            Roots sorted in decreasing order of real part, then imaginary part.
        """
        # Compute roots using NumPy's efficient routine
        roots = np.roots(problem)

        # Sort by real part descending, then imaginary part descending
        # Using a tuple key achieves the required order
        return sorted(roots.tolist(), key=lambda z: (z.real, z.imag), reverse=True)