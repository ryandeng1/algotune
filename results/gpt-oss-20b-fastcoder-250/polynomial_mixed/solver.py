import numpy as np

class Solver:
    def solve(self, problem: list[float], **kwargs) -> list[complex]:
        """
        Compute all roots of the polynomial with coefficients in descending order
        and return them sorted by real part and imaginary part in descending order.
        """
        # Compute roots using numpy's efficient implementation
        roots = np.roots(problem)

        # Sort by real part then imaginary part, descending
        sorted_roots = sorted(roots, key=lambda x: (x.real, x.imag), reverse=True)

        return sorted_roots
