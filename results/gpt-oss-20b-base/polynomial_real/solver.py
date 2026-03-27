import numpy as np
from contextlib import nullcontext
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _single_thread_blas():
    return nullcontext() if threadpool_limits is None else threadpool_limits(limits=1)


class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """
        Solve the polynomial problem by finding all real roots of the polynomial.
        The polynomial is given as a list of coefficients [aₙ, aₙ₋₁, …, a₀].
        The roots are returned as a list of real numbers in descending order.
        """
        # Ensure input is a NumPy array of dtype float for speed.
        coeff = np.asarray(problem, dtype=float)

        with _single_thread_blas():
            roots = np.roots(coeff)

        # Drop negligible imaginary parts and keep only real values.
        # Using tol=1e-3 is efficient and matches the original behavior.
        roots = np.real_if_close(roots, tol=1e-3).astype(float)

        # Sort in descending order – the np.sort operation is already optimal.
        roots.sort()
        roots = roots[::-1]
        return roots.tolist()