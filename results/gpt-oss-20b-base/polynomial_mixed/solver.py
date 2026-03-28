from typing import List
import numpy as np


def _single_thread_blas():
    """
    Stub for the original threadpool limiter. The actual implementation may
    come from `threadpoolctl` or a similar library. For the purpose of these
    benchmarks we simply return a no-op context manager.
    """
    class _DummyContext:
        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover
            return False

    return _DummyContext()


class Solver:
    """
    A lightweight solver that computes all roots of a real‑coefficient
    polynomial and returns them sorted by descending real part (and, as a tiebreaker,
    by descending imaginary part).
    """

    # The method signature is fixed by the specification.
    def solve(self, problem: List[float]) -> List[complex]:
        # Ensure that the coefficient list is not empty and that the leading
        # coefficient is non‑zero. Trim any leading zeros to avoid degenerate
        # cases that np.roots would otherwise handle incorrectly.
        if not problem:
            return []

        # Remove leading zero coefficients to avoid misleading degree inference.
        i = 0
        while i < len(problem) - 1 and abs(problem[i]) < 1e-15:
            i += 1
        coefficients = problem[i:]

        # Delegate the heavy lifting to NumPy, which is highly optimised.
        with _single_thread_blas():
            roots = np.roots(coefficients)

        # Sort by descending real part, then descending imaginary part.
        # Converting to Python primitives speeds up repeated tuple comparisons.
        sorted_roots = sorted(
            roots,
            key=lambda z: (float(z.real), float(z.imag)),
            reverse=True
        )
        return list(sorted_roots)