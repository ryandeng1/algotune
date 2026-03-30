# solver.py
import numpy as np
from numpy.typing import NDArray

class Solver:
    """
    Optimized solver that computes the cumulative integral of a 1‑D array
    using Simpson's rule with a fully vectorised implementation.
    """

    @staticmethod
    def _cumulative_simpson(y: NDArray, dx: float) -> NDArray:
        """
        Compute the cumulative Simpson integral of y with spacing dx.

        Parameters
        ----------
        y : numpy.ndarray
            1‑D array of values.
        dx : float
            Step size along the independent variable.

        Returns
        -------
        numpy.ndarray
            1‑D array containing the cumulative integral at each index.
        """
        n = y.size
        if n < 2:
            return np.zeros(n, dtype=y.dtype)

        # weight vector for composite Simpson's rule:
        # w[0] = 1, w[1] = 4, w[2] = 2, w[3] = 4, w[4] = 2, ...
        # last weight depends on whether the number of intervals is odd or even
        w = np.empty(n, dtype=y.dtype)
        w[0] = 1
        # Create an array of indices 1..n-1
        idx = np.arange(1, n, dtype=int)
        w[idx] = np.where(idx % 2, 4, 2)

        # Adjust the final weight: if number of intervals is odd,
        # the last term is not multiplied by 2 in Simpson's rule.
        if (n - 1) % 2 == 1:  # odd number of intervals
            w[-1] = 1
        else:
            w[-1] = 2

        # Vectorised cumulative sum of weighted values
        cum_integral = np.cumsum(y * w) * dx / 3.0
        return cum_integral

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = np.asarray(problem["y"])
        dx = problem["dx"]
        return self._cumulative_simpson(y, dx)