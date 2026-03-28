import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the input array
        using a vectorised implementation of Simpson's rule. The input
        ``problem`` contains:
        * ``y2``: :class:`numpy.ndarray` – the values to integrate.
        * ``dx``: :class:`float` – the spacing between samples.
        The result will have the same shape as ``y2``.
        """
        y2: NDArray = problem["y2"]
        dx: float = problem["dx"]

        # Ensure the array is contiguous for speed
        y = np.ascontiguousarray(y2)

        # Weight vector for Simpson's rule:
        # w[0] = w[-1] = 1,
        # even indices (except ends): 2,
        # odd indices: 4
        n = y.shape[-1]
        w = np.empty(n, dtype=y.dtype)
        w[:] = 4
        w[0] = 1
        w[-1] = 1
        w[2::2] = 2  # double-check: after setting all to 4, set evens to 2 except ends

        # Cumulative sum of weighted values
        # We need to operate along the last axis
        # Compute weighted array
        weighted = y * w
        # Compute cumulative sum along last axis
        csum = np.cumsum(weighted, axis=-1)

        # Scale by h/3
        result = (dx / 3.0) * csum
        return result