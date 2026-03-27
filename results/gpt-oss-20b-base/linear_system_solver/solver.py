import numpy as np
from typing import Any, List, Dict


class Solver:
    def solve(self, problem: Dict[str, Any]) -> List[float]:
        """
        Solve the linear system Ax = b efficiently using NumPy.
        The implementation validates input shapes,
        ensures contiguous float64 arrays, and delegates to
        the highly‑optimized LAPACK routine for solving.
        """
        # Extract and cast to contiguous float64 arrays
        A = np.asarray(problem["A"], dtype=np.float64, copy=False)
        b = np.asarray(problem["b"], dtype=np.float64, copy=False)

        # Quick shape checks
        n, m = A.shape
        if n != m:
            raise ValueError(f"Matrix A must be square, got shape {A.shape}")
        if b.size != n:
            raise ValueError(f"Vector b size {b.size} does not match A shape {A.shape}")

        # Solve without explicit inversion (uses LAPACK's DSYEV or DGESV)
        x = np.linalg.solve(A, b)

        # Convert result to Python list of floats
        return x.tolist()