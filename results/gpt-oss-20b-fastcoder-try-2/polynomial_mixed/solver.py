import numpy as np
from typing import List

class Solver:
    """
    Fast polynomial root finder using NumPy's efficient implementation.
    The result is sorted in descending order by real part, then imaginary part.
    """
    def solve(self, problem: List[float]) -> List[complex]:
        # NumPy already uses highly optimised LAPACK routines (via OpenBLAS/ATLAS),
        # which are single‑threaded enough for typical workloads. No extra context
        # management is required.
        roots = np.roots(problem)
        # Sort by (real, imag) descending, matching the original requirement.
        return sorted(roots, key=lambda z: (z.real, z.imag), reverse=True)