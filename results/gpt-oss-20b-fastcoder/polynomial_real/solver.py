import numpy as np
from typing import List

def solve(problem: List[float]) -> List[float]:
    """
    Solve the polynomial problem by finding all real roots of the polynomial.

    The polynomial is given as a list of coefficients [aₙ, aₙ₋₁, …, a₀],
    representing:
        p(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + … + a₀.
    This method computes the roots, keeps only the real ones (treating
    small imaginary parts as zero), and gives them sorted in decreasing order.

    :param problem: A list of polynomial coefficients (real numbers) in descending order.
    :return: A list of real roots of the polynomial, sorted in decreasing order.
    """
    # Compute roots with NumPy (uses efficient LAPACK routines)
    roots = np.roots(problem)

    # Discard roots with significant imaginary parts
    roots = np.real_if_close(roots, tol=0.001)

    # Convert to a flat NumPy array of real values
    roots = np.real(roots)

    # Sort in descending order
    roots = np.sort(roots)[::-1]

    return roots.tolist()