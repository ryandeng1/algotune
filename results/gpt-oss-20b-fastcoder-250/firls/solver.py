from contextlib import nullcontext
import numpy as np
from scipy import signal
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

def _single_thread_blas():
    """Limit BLAS to a single thread to avoid oversubscription."""
    return nullcontext() if threadpool_limits is None else threadpool_limits(limits=1)

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        """
        Design an odd-length FIR filter with a passband and stopband defined by
        the provided frequency edges using the least‑squares method.

        Parameters
        ----------
        problem : tuple[int, tuple[float, float]]
            First element is half the filter length (before making it odd).
            Second element is a pair of floats specifying the lower and upper
            edges of the passband, both in the range (0, 1).

        Returns
        -------
        np.ndarray
            1‑D array of real filter coefficients.
        """
        n_half, edges = problem
        # The desired implementation requires an odd length; factor 2*n+1.
        n = 2 * n_half + 1
        edges = tuple(edges)  # Ensure tuple after JSON round‑trip
        # Window specification: passband weight 1, stopband weight 0
        weights = [1, 1, 0, 0]

        with _single_thread_blas():
            coeffs = signal.firls(n, (0.0, *edges, 1.0), weights)

        return coeffs
