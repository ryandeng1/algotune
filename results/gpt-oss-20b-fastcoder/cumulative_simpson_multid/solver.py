from typing import Any
from numpy.typing import NDArray
import numpy as np

class Solver:

    def solve(self, problem: dict) -> NDArray:
        """
        Approximate the cumulative integral of `y2` along the last axis
        using the composite Simpson rule in a vectorised, purely numpy
        implementation (no scipy dependency).

        Parameters
        ----------
        problem: dict
            Must contain ``'y2'`` : NDArray  – the function values
            and ``'dx'``   : float    – uniform spacing of the grid.

        Returns
        -------
        NDArray
            Cumulative integral values with the same shape as ``y2``.
        """
        y2 = problem["y2"]
        dx = problem["dx"]

        # Length of the last dimension
        n = y2.shape[-1]
        if n < 2:
            # Not enough points for Simpson, fallback to trapezoidal
            return np.cumsum(y2 * dx, axis=-1) - y2[...,0] * dx

        # Create weights for Simpson rule: 1,4,2,4,...,4,1
        weights = np.empty(n, dtype=y2.dtype)
        weights[0] = 1
        weights[1::2] = 4
        weights[2::2] = 2
        # correct the last weight: if the last point uses coefficient 1
        if n % 2 == 0:
            weights[-1] = 1
        else:
            # odd number of points -> last point coefficient is 1 too
            weights[-1] = 1

        # Compute weighted cumulative sum along the last axis
        weighted = y2 * weights
        cumsum = np.cumsum(weighted, axis=-1)

        # The cumulative integral starts at zero; shift by one index
        result = np.empty_like(y2)
        result[..., 0] = 0
        result[..., 1:] = (dx / 3.0) * cumsum[..., :-1]

        return result