import numpy as np
from contextlib import nullcontext

# When available, limit BLAS threads to 1 (avoids race conditions on some systems)
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _single_thread_blas():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[complex]:
        """Return all roots of the polynomial given by ``problem`` (coefficients
        from highest degree to constant term). The roots are sorted in descending
        order first by real part and then by imaginary part."""
        # Convert to a NumPy array; this triggers a fast C implementation
        coeffs = np.asarray(problem, dtype=float)

        # Compute roots (this is the heavy part)
        with _single_thread_blas():
            roots = np.roots(coeffs)

        # Sort in descending order by real then imag parts
        # np.lexsort expects keys from last to first
        order = np.lexsort((roots.imag, roots.real))
        sorted_roots = roots[order[::-1]].tolist()
        return sorted_roots