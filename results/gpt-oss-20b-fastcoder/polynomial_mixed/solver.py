import os
import numpy as np
from typing import List

# Disable multithreading in the BLAS backend once for the lifetime of the program
# (works for OpenBLAS, MKL, and other common libraries).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

class Solver:
    def solve(self, problem: List[float]) -> List[complex]:
        """
        Find all roots of the polynomial defined by `problem` and return them sorted
        in descending order by real part and, secondarily, by imaginary part.
        """
        # numpy.roots uses LAPACK and may use multiple threads; the environment
        # variables set above ensure single–threaded operation.
        roots = np.roots(problem)

        # Sort by descending real part, then descending imaginary part.
        # np.lexsort sorts by the last key first; for descending order we negate.
        # Key 0 (last) : real part, key 1 : imag part.
        idx = np.lexsort((-roots.imag, -roots.real))
        sorted_roots = roots[idx]
        return sorted_roots.tolist()