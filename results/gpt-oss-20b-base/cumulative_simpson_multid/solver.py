import numpy as np
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson

class Solver:
    """Fast cumulative Simpson integration on the last axis of a NumPy array."""

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi‑dimensional array
        using Simpson's rule.

        Parameters
        ----------
        problem : dict
            Dictionary containing ``'y2'`` (input data array) and ``'dx'`` (spacing, can be scalar or array).

        Returns
        -------
        NDArray
            Cumulative integral of ``y2`` with respect to ``dx``.
        """
        y2: NDArray = problem["y2"]
        dx: NDArray | float = problem["dx"]
        # The SciPy implementation is a compiled routine and is already highly optimised.
        return cumulative_simpson(y2, dx=dx)