import numpy as np
from contextlib import nullcontext

# Optional thread polling limit from the original snippet
try:
    from threadpoolctl import threadpool_limits  # type: ignore
except Exception:
    threadpool_limits = None

def _single_thread_blas():
    return nullcontext() if threadpool_limits is None else threadpool_limits(limits=1)

class Solver:
    """
    Fast polynomial root solver.
    The implementation aims to reduce overhead for small to medium degree polynomials
    by:
      * Using ``numpy.polynomial.Polynomial`` which performs a single eigenvalue
        computation via the companion matrix (the same as ``np.roots`` but with a
        small overhead reduction).
      * Avoiding the costly ``np.real_if_close`` by a simple tolerance check.
      * Sorting with a fast in‑place reverse of the array.
    """
    def solve(self, problem: list[float]) -> list[float]:
        coeffs = np.asarray(problem, dtype=np.float64)

        # Ensure we treat leading zeros uniformly
        if coeffs.size > 1 and coeffs[0] == 0:
            # Strip leading zeros; if all zero, return empty list
            nonzero_idx = np.nonzero(coeffs)[0]
            if nonzero_idx.size == 0:
                return []
            coeffs = coeffs[nonzero_idx[0]:]

        # Compute roots via the companion matrix method
        with _single_thread_blas():
            # scipy's roots implementation is essentially the same but
            # numpy's polynomials are slightly faster for small arrays
            from numpy.polynomial import Polynomial
            p = Polynomial(coeffs[::-1])  # Polynomial expects low→high
            roots = p.roots()

        # Filter out spurious imaginary parts
        tol = 1e-3
        imag = np.abs(roots.imag)
        real_mask = imag <= tol
        real_roots = roots[real_mask].real

        # Sort in decreasing order
        real_roots.sort()
        real_roots = real_roots[::-1]

        return real_roots.tolist()