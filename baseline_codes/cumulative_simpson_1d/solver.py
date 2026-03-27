from typing import Any
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson


class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        """
        y = problem["y"]
        dx = problem["dx"]
        result = cumulative_simpson(y, dx=dx)
        return result
