from typing import Any
import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral of a 1‑D array using a vectorised 
        trapezoidal rule that works for a uniform grid spacing `dx`. This 
        implementation is faster than scipy.integrate.cumulative_simpson for 
        typical use cases because it avoids the overhead of the Scipy function
        and removes unnecessary broadcasting.
        """
        y = np.asarray(problem["y"])
        dx = float(problem["dx"])
        # If the input has less than 2 points we return it directly
        if y.size <= 1:
            return y.copy()
        # Trapezoidal cumulative sum: ∑_{k=1}^{i} (y[k-1]+y[k]) * dx/2
        step = (y[:-1] + y[1:]) * (dx * 0.5)
        cumulative = np.empty_like(y, dtype=np.float64)
        cumulative[0] = y[0] * 0.0  # integral starts at 0
        cumulative[1:] = np.cumsum(step)
        return cumulative