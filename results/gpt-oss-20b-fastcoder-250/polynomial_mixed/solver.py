import numpy as np
from contextlib import nullcontext
from typing import List, Iterable

# Attempt to import thread limiting; fall back to nullcontext if unavailable.
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _single_thread_blas() -> Iterable[None]:
    """Limit NumPy BLAS to a single thread during kernel calls."""
    return threadpool_limits(limits=1) if threadpool_limits else nullcontext()


class Solver:
    def solve(self, problem: List[float]) -> List[complex]:
        """
        Return all roots of a real‑coefficient polynomial sorted by decreasing
        real part, then imaginary part.
        """
        coeffs = np.asarray(problem, dtype=float)
        # Guard against zero polynomial or leading zeros.
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise ValueError("Polynomial coefficients must be a non‑empty 1‑D array.")
        # Drop leading zeros to avoid degenerate roots.
        first_non_zero = np.nonzero(coeffs)[0]
        if len(first_non_zero) == 0:
            return []
        coeffs = coeffs[first_non_zero[0] :]

        with _single_thread_blas():
            roots = np.roots(coeffs)

        # Sort: highest real part first, then highest imag part.
        roots.sort(key=lambda z: (z.real, z.imag), reverse=True)
        return roots.tolist()