import numpy as np
from typing import Any, List

class Solver:
    """
    Optimised linear system solver.

    Uses :pyfunc:`numpy.linalg.solve` for square systems, which is highly optimised
    in the underlying LAPACK libraries.  For non‑square systems it falls back to a
    least‑squares solution with :pyfunc:`numpy.linalg.lstsq`.  Input conversion
    is performed with :pyfunc:`numpy.asarray` to avoid unnecessary copies.
    """
    def solve(self, problem: dict[str, Any]) -> List[float]:
        A = np.asarray(problem["A"])
        b = np.asarray(problem["b"])

        # Ensure correct dtype – LAPACK works best with float64
        if A.dtype != np.float64:
            A = A.astype(np.float64, copy=False)
        if b.dtype != np.float64:
            b = b.astype(np.float64, copy=False)

        # Square matrix ⇒ direct solve, otherwise least‑squares
        if A.ndim == 2 and A.shape[0] == A.shape[1]:
            x = np.linalg.solve(A, b)
        else:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)

        return x.tolist()