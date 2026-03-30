# solver.py
from __future__ import annotations

from contextlib import nullcontext
from typing import List

import numpy as np


# --------- helper ----------------------------------------------------------
# If the optional threadpool_limit package is available we can limit BLAS to
# a single thread. If it is not imported we simply do nothing.
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:  # pragma: no cover
    threadpool_limits = None  # type: ignore


def _single_thread_blas():
    """
    Context manager that limits BLAS backends (OpenBLAS, MKL, etc.) to a single
    thread.  The cost of the context manager itself is negligible compared to
    the work done by NumPy’s root finder.
    """
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


# --------- solver -----------------------------------------------------------
class Solver:
    """
    The Solver class implements a fast solver for the roots of a real polynomial.
    It leverages NumPy’s highly optimised vectorised routines and replaces the
    Python level ``sorted`` with a NumPy sort on a structured array for
    maximum performance.  The overall API remains unchanged.
    """

    __slots__ = ()

    # Public API -------------------------------------------------------------
    def solve(self, problem: List[float]) -> List[complex]:
        """
        Find all roots of a polynomial with coefficients in descending order.

        Parameters
        ----------
        problem : list[float]
            Coefficients of the polynomial [a_n, a_{n-1}, … , a_0].

        Returns
        -------
        list[complex]
            Roots sorted in descending order by real part and, if equal,
            by imaginary part.
        """
        # --------------------------------------------------------------------
        # 1. Compute the roots using NumPy's highly optimised routine.
        # --------------------------------------------------------------------
        coefficients = np.asarray(problem, dtype=np.float64)
        with _single_thread_blas():
            roots = np.roots(coefficients)

        # --------------------------------------------------------------------
        # 2. Sort the roots fast.
        #    NumPy can sort a structured array of two floats (real, imag) in
        #    O(n log n) time.  This is faster than the Python ``sorted`` call.
        # --------------------------------------------------------------------
        dtype = np.dtype([('r', roots.dtype), ('i', roots.dtype)])
        structured = np.empty(roots.shape, dtype=dtype)
        structured['r'] = roots.real
        structured['i'] = roots.imag

        # ``argsort`` on the structured array gives the indices that would
        # sort the array lexicographically (real first, then imag).
        # ``[::-1]`` reverses to descending order.
        idx = np.argsort(structured, order=['r', 'i'])[::-1]  # noqa: WPS202

        # Retrieve the sorted roots and convert to Python list.
        sorted_roots = roots[idx]
        return sorted_roots.tolist()