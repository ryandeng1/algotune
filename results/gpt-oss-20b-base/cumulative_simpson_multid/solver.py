import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis using a
        vectorised Simpson's rule implementation. This avoids the
        slower pure‑python loops inside scipy's `cumulative_simpson` and
        works for any array shape.
        """
        y2: NDArray = problem["y2"]
        dx: float | NDArray = problem["dx"]

        # Ensure dx is a scalar or an array that can broadcast to y2.shape
        dx = np.asarray(dx)
        if dx.ndim == 0:
            dx = float(dx)

        # Number of points along the last axis
        n = y2.shape[-1]

        # Pre‑allocate result array
        result = np.empty_like(y2)

        # Simpson's rule requires an even number of intervals -> odd number of points.
        # If n is even we drop the last point to keep an odd number of points.
        if n % 2 == 0:
            y2 = y2[..., :-1]
            n -= 1

        # Weights for Simpson's rule: 1/3, 4/3, 2/3, 4/3, …, 1/3
        # Create a weights array of shape (n,)
        weights = np.empty(n, dtype=y2.dtype)
        weights[0] = 1.0 / 3.0
        weights[-1] = 1.0 / 3.0
        weights[1:-1:2] = 4.0 / 3.0   # odd indices (starting at 1)
        weights[2:-2:2] = 2.0 / 3.0   # even indices (excluding ends)

        # Expand weights to match y2's shape for broadcasting
        # Ensure last axis is the one we multiply
        for _ in range(y2.ndim - 1):
            weights = np.expand_dims(weights, axis=0)

        # Weighted sum along the last axis gives the cumulative integral
        # We use a cumulative sum of the weighted values * dx
        weighted = y2 * weights
        # Cumulative sum along last axis
        result = np.cumsum(weighted, axis=-1) * dx

        # If we dropped a point earlier, pad the result with the last value
        if n == y2.shape[-1] + 1:
            # Append the last cumulative value along the last axis
            last_col = result[..., -1][..., None]
            result = np.concatenate([result, last_col], axis=-1)

        return result