import numpy as np


class Solver:
    """Find all roots of a real‑coefficient polynomial and return them sorted
    in descending order by real part and, secondarily, by imaginary part.
    """

    def solve(self, problem: list[float]) -> list[complex]:
        """
        Parameters
        ----------
        problem : list[float]
            Polynomial coefficients in descending order.

        Returns
        -------
        list[complex]
            All roots sorted in descending order by real part, then by imaginary part.
        """
        # Use NumPy's robust polynomial root finder
        roots = np.roots(problem)

        # Sort in descending order by real, then imaginary part
        # np.lexsort uses the last key as the primary; we negate to get descending order
        idx = np.lexsort((-roots.imag, -roots.real))
        sorted_roots = roots[idx]

        # Convert back to Python list for the expected return type
        return sorted(sorted_roots, key=lambda z: (z.real, z.imag), reverse=True)