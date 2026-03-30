# solver.py
from __future__ import annotations

from typing import List

import numpy as np
from contextlib import nullcontext

# -------------------------------------------------------------
# Helper to keep NumPy BLAS single‑threaded (no external deps)
# -------------------------------------------------------------
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

def _single_thread_blas():
    """Context manager that limits any NumPy BLAS thread pool to a single thread."""
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)

# -------------------------------------------------------------
# Solver implementation
# -------------------------------------------------------------
class Solver:
    """
    Find all real roots of a real‑coefficient polynomial.

    The polynomial is supplied as a list of coefficients in descending order.
    Roots are returned sorted in decreasing order.
    """

    def solve(self, problem: List[float]) -> List[float]:
        """
        Parameters
        ----------
        problem:
            List of coefficients `[a_n, a_{n-1}, ..., a_0]` describing
            the polynomial `a_n*x^n + a_{n-1}*x^{n-1} + ... + a_0`.

        Returns
        -------
        List[float]
            Real roots sorted in descending order.
        """
        # Compute the roots – NumPy takes care of the heavy lifting
        with _single_thread_blas():
            roots = np.roots(problem)

        # Drop roots that are effectively real
        # `real_if_close` uses 2**-x tolerance internally; 0.001 forces removal
        roots = np.real_if_close(roots, tol=0.001)

        # Take the real part – any tiny imaginary remains are discarded
        roots = np.real(roots)

        # Sort in decreasing order in a single pass
        roots.sort()
        roots = roots[::-1]

        return roots.tolist()