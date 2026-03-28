import numpy as np
from contextlib import nullcontext

# Optional: disable multithreading of BLAS (fallback to single thread if available)
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

def _single_thread_blas():
    return threadpool_limits(limits=1) if threadpool_limits else nullcontext()


class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """Return the roots of a polynomial in descending order by real part,
        then by imaginary part.

        Parameters
        ----------
        problem : list[float]
            Polynomial coefficients in descending order.
        """
        coeffs = np.asarray(problem, dtype=float)
        # Guard against zero polynomial
        if coeffs.size == 0:
            return []

        with _single_thread_blas():
            roots = np.roots(coeffs)

        # Sort by real part descending, then by imaginary part descending
        # Using numpy for speed
        indices = np.lexsort(( -roots.imag, -roots.real ))
        return roots[indices].tolist()