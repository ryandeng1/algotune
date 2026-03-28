import numpy as np
from contextlib import nullcontext

def _single_thread_blas():
    # In this environment we do not control BLAS threads; return a no-op context.
    return nullcontext()

class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """
        Find all roots of a univariate polynomial with real coefficients.

        Parameters
        ----------
        problem : list[float]
            Coefficients of the polynomial in descending order of powers,
            e.g. [a_n, a_{n-1}, ..., a_0] for p(x) = a_n x^n + … + a_0.

        Returns
        -------
        list[complex]
            Roots sorted descending by real part, then by imaginary part.
        """
        # Compute roots using NumPy (C implemented).
        with _single_thread_blas():
            roots = np.roots(problem)

        # Sort lexicographically: first by real part, then by imaginary part,
        # both in descending order.  We use numpy's lexsort on the negative
        # values to avoid Python loops and achieve speed.
        # Since lexsort sorts with the last key as primary, we provide the
        # keys in reverse order (imag, real).
        key_real  = -roots.real
        key_imag  = -roots.imag
        idx_sorted = np.lexsort((key_imag, key_real))
        sorted_roots = roots[idx_sorted]

        # Convert to Python list of complex numbers.
        return sorted_roots.tolist()