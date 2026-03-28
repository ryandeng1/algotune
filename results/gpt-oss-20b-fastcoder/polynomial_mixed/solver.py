import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Find all complex roots of the polynomial whose coefficients are given
        in descending order and return them sorted in descending order by
        real parts, then by imaginary parts.
        """
        # Compute the roots using NumPy's highly optimized routine
        roots = np.roots(problem)

        # Sort by real part descending, then by imaginary part descending.
        # np.lexsort sorts by the last key first, so we reverse the keys.
        sorted_idx = np.lexsort((roots.imag, roots.real))[::-1]
        return roots[sorted_idx].tolist()