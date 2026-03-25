# solver.py
from contextlib import nullcontext
import numpy as np
from scipy import signal
from threadpoolctl import threadpool_limits


def _single_thread_blas():
    """Ensure single-threaded BLAS for deterministic results."""
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem, **kwargs):
        """
        Design an odd-length FIR filter using the least-squares method (firls).
        Parameters
        ----------
        problem : tuple[int, tuple[float, float]]
            (n, (low_edge, high_edge)) where n is the half-length
            and edges define the passband.
        Returns
        -------
        np.ndarray
            The filter coefficients of length 2*n+1.
        """
        n, edges = problem
        # Ensure filter length is odd
        M = 2 * n + 1
        # Convert edges to tuple if needed
        edges = tuple(edges)

        # Construct desired amplitude vector: pass, pass, stop, stop
        # and breakpoint frequencies: 0.0, low_edge, high_edge, 1.0
        with _single_thread_blas():
            coeffs = signal.firls(M, (0.0, *edges, 1.0), [1, 1, 0, 0])

        return coeffs
