import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem: dict) -> np.ndarray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array using Simpson's rule.
        The implementation delegates to scipy's optimized C routine which is vectorised and fast.
        """
        y2 = problem["y2"]
        dx = problem["dx"]
        # scipy's cumulative_simpson handles arbitrary shaped arrays with
        # the last axis being integrated.
        return cumulative_simpson(y2, dx=dx)
