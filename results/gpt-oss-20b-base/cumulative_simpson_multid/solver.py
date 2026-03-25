import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict, **kwargs) -> NDArray:
        """
        Compute the cumulative integral along the last axis of a multi-dimensional array
        using Simpson's rule, but implemented in a fully vectorised way for speed.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
            - "y2": 3‑D numpy array of shape (100, 100, 1000) containing sample values.
            - "dx": float, spacing between successive sample points.

        Returns
        -------
        NDArray
            Array of the same shape with the cumulative Simpson integral.
        """
        y = problem["y2"]
        dx = problem["dx"]

        # length of the last axis
        M = y.shape[-1]
        # Pre–compute Simpson weights for a fixed M
        weights = np.empty(M, dtype=y.dtype)
        weights[0] = 1
        if M > 1:
            weights[1] = 4
        if M > 2:
            # interior pattern: 2,4 repeated
            weights[2: M - 1: 2] = 2
            weights[3: M - 1: 2] = 4
            weights[-1] = 1

        # Weighted cumulative sum
        weighted = y * weights
        cum = np.cumsum(weighted, axis=-1) * (dx / 3.0)

        # Simpson's rule requires at least two intervals to produce a non‑zero integral.
        # The first two entries are set to zero.
        cum[..., 0] = 0
        if M > 1:
            cum[..., 1] = 0

        return cum
