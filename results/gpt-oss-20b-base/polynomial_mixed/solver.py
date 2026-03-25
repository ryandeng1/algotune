import numpy as np
from contextlib import nullcontext

# In case threadpool_limits is unavailable, define a dummy context manager
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

def _single_thread_blas():
    return nullcontext() if threadpool_limits is None else threadpool_limits(limits=1)

class Solver:
    def solve(self, problem: list[float], **kwargs) -> list[complex]:
        """
        Compute all roots of a real-coefficient polynomial and return them
        sorted in descending order by real part and then by imaginary part.

        Parameters
        ----------
        problem : list[float]
            Coefficients of the polynomial in descending order [a_n, ..., a_0].

        Returns
        -------
        list[complex]
            Sorted list of roots.
        """
        coeffs = problem
        # Ensure only one thread is used for LAPACK operations
        with _single_thread_blas():
            roots = np.roots(coeffs)
        # Sort by (real, imag) descending
        roots_sorted = sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)
        return roots_sorted
