import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis using the composite Simpson rule.
        This manual implementation avoids the overhead of ``scipy.integrate.cumulative_simpson``.
        """
        y = np.asarray(problem["y2"])
        dx = float(problem["dx"])

        # Number of points along the last axis
        n = y.shape[-1]
        if n < 3:
            # Not enough points for Simpson; fallback to trapezoidal
            return np.cumsum(y, axis=-1) * dx

        # Pre‑compute weights for Simpson's rule
        # For points 0 and n-1 use 1/6*dx, others use 4/6*dx or 2/6*dx
        coeff = np.empty(n, dtype=y.dtype)
        coeff[:] = 2.0
        coeff[0] = coeff[-1] = 1.0
        coeff[1::2] = 4.0  # odd indices
        coeff *= dx / 6.0

        # Cumulative integral: sum of y * coeff across the last axis
        # Use np.cumsum on the weighted array
        weighted = y * coeff
        return np.cumsum(weighted, axis=-1)