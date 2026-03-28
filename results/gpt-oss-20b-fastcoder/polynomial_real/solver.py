import numpy as np
from contextlib import nullcontext
from typing import List

# ------------------------------------------------------------------------------
# Helper to keep a single thread for BLAS routines (used only if available)
# ------------------------------------------------------------------------------

try:
    from numba import threadpool_limits  # type: ignore
except Exception:
    threadpool_limits = None

def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)

# ------------------------------------------------------------------------------
# Solver implementation
# ------------------------------------------------------------------------------

class Solver:
    """
    Solves for all real roots of a polynomial given by its coefficients in
    descending order.
    """

    def solve(self, problem: List[float]) -> List[float]:
        """
        Find all real roots of the polynomial defined by *problem*.

        Parameters
        ----------
        problem : list[float]
            Polynomial coefficients [aₙ, aₙ₋₁, ..., a₀].

        Returns
        -------
        list[float]
            Sorted list of real roots in descending order.
        """
        # Compute roots once in a single‑threaded BLAS environment.
        with _single_thread_blas():
            coeffs = np.asarray(problem, dtype=float)
            roots = np.roots(coeffs)

        # Keep only roots with negligible imaginary part.
        imag_tol = 1e-7  # tighter tolerance – faster than using np.real_if_close
        real_roots = np.real(roots[np.abs(np.imag(roots)) <= imag_tol])

        # Sort in descending order and convert to Python list.
        return np.sort(real_roots, kind="quicksort")[::-1].tolist()