import numpy as np
from numpy.typing import NDArray

class Solver:

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi-dimensional array
        using a fast vectorized trapezoidal rule.
        """
        y2 = np.asarray(problem['y2'])
        dx = problem['dx']
        # use numpy's cumulative trapezoidal integration
        # np.cumtrapz returns an array of the same shape as y2
        result = np.cumtrapz(y2, dx=dx, axis=-1, initial=0)
        return result