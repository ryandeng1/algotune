import numpy as np
from scipy import signal
from contextlib import nullcontext

# Optional dependency for limiting BLAS threads
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

def _single_thread_blas():
    """Context manager to limit BLAS to a single thread."""
    return threadpool_limits(limits=1) if threadpool_limits else nullcontext()

class Solver:
    def solve(self, problem: tuple[int, tuple[float, float]]) -> np.ndarray:
        """
        Design an FIR low‑pass filter using scipy.signal.firls.

        Parameters
        ----------
        problem : tuple[int, tuple[float, float]]
            * ``n`` : Half‑length of the desired filter (int).
            * ``edges`` : (low_cut, high_cut) band edges as a tuple.

        Returns
        -------
        coeffs : np.ndarray
            The symmetric filter coefficients.
        """
        n, edges = problem
        # Ensure filter length is odd as required by firls.
        n = 2 * n + 1
        low_cut, high_cut = edges

        # Limited parallelism makes the call deterministic for small n.
        with _single_thread_blas():
            coeffs = signal.firls(
                n,
                (0.0, low_cut, high_cut, 1.0),
                [1, 1, 0, 0],
            )
        return coeffs