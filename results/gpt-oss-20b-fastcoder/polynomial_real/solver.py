import numpy as np
from contextlib import nullcontext

# Detected if threadpool_limits is available; keep it optional
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _single_thread_blas():
    """Ensure BLAS works in single‑thread mode."""
    return nullcontext() if threadpool_limits is None else threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """
        Compute all real roots of a polynomial given by its coefficients.

        Parameters
        ----------
        problem : list[float]
            Coefficients [a_n, a_{n-1}, …, a_0] defining the polynomial

        Returns
        -------
        list[float]
            Real roots sorted in decreasing order.
        """
        coeff = problem  # local alias
        with _single_thread_blas():
            roots = np.roots(coeff)

        # Discard roots with significant imaginary parts
        img = np.imag(roots)
        if np.any(np.abs(img) > 0.001):
            roots = roots[np.abs(img) <= 0.001].real
        else:
            roots = roots.real

        # Sort descending
        roots.sort()
        return roots[::-1].tolist()