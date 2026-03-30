import numpy as np
from typing import List

# --------------------------------------------------------------------
# Solver class
# --------------------------------------------------------------------

class Solver:
    """
    Solver for finding all real roots of a polynomial.
    """

    @staticmethod
    def solve(problem: List[float]) -> List[float]:
        """
        Compute the real roots of a polynomial given by its coefficients.

        Args:
            problem: List of coefficients [aₙ, aₙ₋₁, …, a₀] for
                     p(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + … + a₀.

        Returns:
            Sorted (descending) list of real roots.
        """
        # Fast path: if the polynomial is linear, solve analytically
        deg = len(problem) - 1
        if deg == 1:
            return [-(problem[1] / problem[0])]

        # Compute complex roots using numpy's efficient LAPACK routine
        roots = np.roots(problem)

        # Treat very small imaginary parts as zero (tolerance 1e-3)
        roots = np.real_if_close(roots, tol=0.001)

        # Keep only the real part
        roots = np.real(roots)

        # Sort descending
        roots.sort()
        roots = roots[::-1]

        return roots.tolist()