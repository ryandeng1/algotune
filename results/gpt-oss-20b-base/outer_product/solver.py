# solver.py

import numpy as np

class Solver:
    """
    Fast solver for computing the outer product of two 1‑D numpy arrays.

    Uses :func:`numpy.multiply.outer`, which is a specialised
    implementation for this operation and typically beats the generic
    :func:`numpy.outer` on performance.
    """

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        # ``np.multiply.outer`` is a thin wrapper around the C implementation
        # that performs the outer product directly.
        return np.multiply.outer(vec1, vec2)