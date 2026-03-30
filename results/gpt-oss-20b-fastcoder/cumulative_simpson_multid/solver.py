# solver.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Optimised cumulative Simpson integrator.

    Computes the cumulative integral along the last axis of a NumPy array using
    Simpson's rule.  The implementation uses fully vectorised operations and
    avoids the Python‐level loop of :func:`scipy.integrate.cumulative_simpson`,
    resulting in a significant speed‑up for large arrays.

    The recurrence for Simpson’s rule on an array `y` with constant step `dx`
    is:

        I[k] = I[k-1] + dx/3 * (w[k-1]*y[k-1] + w[k]*y[k]),

    where the weights `w` alternate 1,4,2,4,2,… (with the last weight being 1
    if the number of intervals is odd).  By pre‑computing the cumulative sum of
    the weighted samples we can obtain all partial integrals in a single
    `np.cumsum` call.

    Parameters
    ----------
    problem : dict
        Must contain the keys:
            * 'y2'   : NDArray, the function values
            * 'dx'   : float,   uniform spacing between samples

    Returns
    -------
    NDArray
        Cumulative integral array of the same shape as `y2`.
    """

    def __init__(self) -> None:
        # Nothing to pre‑compute for the generic case.
        pass

    # --------------------------------------------------------------------
    def solve(self, problem: dict) -> NDArray:
        y2: NDArray = problem["y2"]
        dx: float = problem["dx"]

        # Ensure `y2` is at least 1‑D for slice handling.
        if y2.ndim == 0:
            # Scalar case: integral is simply the value times the step.
            return np.array([0.0, dx * y2]) if y2.shape == () else np.zeros_like(y2)

        last_dim = y2.shape[-1]

        # Allocate weight array: 1,4,2,4,2,…,1 (if odd number of intervals).
        # For even number of intervals the last weight is 2.
        w = np.empty(last_dim, dtype=y2.dtype)
        w[0] = 1
        if last_dim > 1:
            # Even indices (1‑based) get weight 4, odd get 2.
            w[1:] = 4  # start with 4 for index=1
            w[2::2] = 2  # override even indices (2,4,…) with 2

        # For an odd number of intervals the last weight should be 1.
        if last_dim % 2 == 0:
            w[-1] = 2
        else:
            w[-1] = 1

        # Compute weighted cumulative sum along the last axis.
        weighted = y2 * w
        cum_weighted = np.cumsum(weighted, axis=-1, dtype=y2.dtype)

        # Apply the Simpson prefactor.
        result = (dx / 3) * cum_weighted

        return result