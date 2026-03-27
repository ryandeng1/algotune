import numpy as np
from contextlib import nullcontext

# If available, force single‑thread BLAS; otherwise, no-op.  We avoid external imports.
try:
    from threadpoolctl import threadpool_limits
except Exception:
    def _single_thread_blas():
        return nullcontext()
else:
    def _single_thread_blas():
        return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """
        Compute all real roots of a polynomial with coefficients in descending order.
        The result is sorted in descending order.
        """
        with _single_thread_blas():
            roots = np.roots(problem)

        # Filter away roots with significant imaginary parts
        roots = np.real_if_close(roots, tol=1e-3)
        roots = np.real(roots)

        # Return sorted list (descending)
        return np.sort(roots)[::-1].tolist()