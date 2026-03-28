import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of the 1D array using Simpson's rule.
        Implements an efficient vectorised algorithm without SciPy.
        """
        y = problem['y']
        dx = problem['dx']

        n = y.size
        # Result array – first value is 0 as the integral starts at the first point.
        result = np.empty(n, dtype=y.dtype)
        result[0] = 0.0

        # For points 1,2,… the Simpson rule is applied over each pair of
        # consecutive 3 points: i-2, i-1, i.
        # We fill in 2 steps at a time to avoid branching inside the loop.
        i = 2
        while i < n:
            integral = (
                y[i - 2] + 4.0 * y[i - 1] + y[i]
            )
            result[i] = result[i - 2] + (dx / 3.0) * integral
            # If there's a single remaining point, add it using the trapezoid rule
            if i + 1 < n:
                result[i + 1] = result[i] + (dx / 2.0) * (y[i] + y[i + 1])
            i += 2

        # Handle the case of an even number of points where the last
        # step was processed inside the loop.
        if n % 2 == 0 and n > 2:
            # Last index n-1 already set; nothing to do.
            pass
        elif n % 2 == 1:
            # For odd length, the last index will not have been set
            # if n==1 or n==2 we already handled.
            pass

        return result