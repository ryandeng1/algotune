import numpy as np

class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """
        Find all real roots of a polynomial with real coefficients and
        return them sorted in descending order.  The input `problem`
        is a list of coefficients in descending order.
        """
        # Compute roots using numpy's efficient polynomial solver
        roots = np.roots(problem)

        # Discard any roots with large imaginary part
        # np.real_if_close keeps roots whose imaginary part is < 1e-3 * max(|real|,1)
        roots = np.real_if_close(roots, tol=1e-3)

        # Convert to real numbers and sort in descending order
        real_roots = np.real(roots)
        sorted_roots = np.sort(real_roots)[::-1]

        # Round to thousandths to ensure 0.001 precision
        rounded = np.round(sorted_roots, 3)

        return rounded.tolist()
