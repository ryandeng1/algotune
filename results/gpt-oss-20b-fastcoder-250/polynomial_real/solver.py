import numpy as np
from contextlib import nullcontext

try:
    from threadpoolctl import threadpool_limits
    _threadpool_available = True
except Exception:
    _threadpool_available = False

def _single_thread_blas():
    if not _threadpool_available:
        return nullcontext()
    return threadpool_limits(limits=1)

class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """Compute all real roots of a polynomial with real coefficients.

        Parameters
        ----------
        problem : list[float]
            Polynomial coefficients in descending order.

        Returns
        -------
        list[float]
            Real roots sorted in descending order.
        """
        coeffs = np.asarray(problem, dtype=np.float64)
        with _single_thread_blas():
            roots = np.roots(coeffs)
        # Make negligible imag parts real
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.real(roots)
        # Sort descending
        roots = np.sort(roots)[::-1]
        return roots.tolist()
