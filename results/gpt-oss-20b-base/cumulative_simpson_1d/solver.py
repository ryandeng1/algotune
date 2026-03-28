import numpy as np
from typing import Any
from numpy.typing import NDArray


class Solver:
    """Fast cumulative Simpson integrator using NumPy."""

    def solve(self, problem: dict) -> NDArray:
        """Return cumulative integral of y with spacing dx using Simpson's rule."""
        y = np.asarray(problem["y"])
        dx = float(problem["dx"])
        n = y.size
        if n < 2:
            return np.empty(n, dtype=y.dtype)

        # Weight pattern 1,4,2,4,...,4,1 for Simpson
        # Create array of weights
        w = np.empty(n, dtype=y.dtype)
        w[0] = w[-1] = 1
        # Internals: even indices (1-based) get 4, odd get 2
        # For 0-based indexing: indices 1,3,5,... (odd) get 4, even (2,4,6,...) get 2
        w[1:-1:2] = 4
        w[2:-1:2] = 2

        # Cumulative weighted sum
        cum = np.cumsum(w * y)
        return (dx / 3.0) * cum