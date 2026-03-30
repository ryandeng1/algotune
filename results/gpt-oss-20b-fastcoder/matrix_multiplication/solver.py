from typing import Any
import numpy as np

class Solver:
    """
    Optimised solver for matrix multiplication.

    Notes on optimisation:
    * We use `numpy.matmul` which dispatches to highly tuned BLAS routines on most
      platforms.  It is slightly faster than `numpy.dot` for typical 2‑D array
      multiplication in recent NumPy versions.
    * The inputs are converted to `float64` arrays once, ensuring that the BLAS
      routine receives data in a contiguous block.  This is the fastest path for
      large matrices.  Converting to a lower precision would reduce accuracy
      and is not generally beneficial for a pure multiplication problem.
    * Return the result as a Python list of lists to satisfy the required
      interface – the conversion is cheap compared to the multiplication itself
      and keeps the public API unchanged.
    """
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        # Convert the input lists to contiguous float64 arrays
        A = np.asarray(problem['A'], dtype=np.float64, order='C')
        B = np.asarray(problem['B'], dtype=np.float64, order='C')

        # Perform matrix multiplication via BLAS
        C = A @ B  # equivalent to np.matmul(A, B)

        # Convert back to a list of lists
        return C.tolist()