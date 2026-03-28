import numpy as np
from contextlib import nullcontext

def _single_thread_blas():
    # Assume no external thread pool limiter; return a no-op context manager.
    return nullcontext()

class Solver:
    def solve(self, problem):
        """
        Solve a polynomial with real coefficients.

        Parameters
        ----------
        problem : list[float]
            Coefficients [aₙ, aₙ₋₁, …, a₀] of the polynomial aₙxⁿ + … + a₀.

        Returns
        -------
        list[float]
            Sorted real roots (descending order). Roots with
            negligible imaginary part are cast to real.
        """
        # Convert to numpy array once
        coeffs = np.array(problem, dtype=float)
        # Avoid multi‑threaded BLAS for deterministic timing
        with _single_thread_blas():
            roots = np.roots(coeffs)
        # Treat small imaginary parts as numerical noise
        roots = np.real_if_close(roots, tol=1.0 / 1024)
        # Extract real part and sort descending
        real_roots = np.sort(roots.real)[::-1]
        return real_roots.tolist()