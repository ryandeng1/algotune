import numpy as np
from typing import Any
from numpy.typing import NDArray

class Solver:
    @staticmethod
    def _cumulative_simpson(x: NDArray, dx: float) -> NDArray:
        """
        Compute the cumulative integral of `x` along its last axis using Simpson's rule.

        Parameters
        ----------
        x : NDArray
            1‑D or multi‑dimensional input array.
        dx : float
            Uniform grid spacing.

        Returns
        -------
        NDArray
            Cumulative integral along the last dimension.
        """
        # Ensure torchlike broadcasting
        # Permute so the last axis is axis 0 for vectorised computation
        x = np.swapaxes(x, -1, 0)
        n = x.shape[0]

        # For a single point we just return zeros
        if n == 1:
            return np.zeros_like(x)

        # Wrap the first `n` samples to compute partial sums of Simpson's rule
        # Append the last sample to close interval
        extended = np.concatenate((x, x[-1:]), axis=0)

        # weights: 1,4,2,4,2,... (first and last treat specially)
        # Use integer arithmetic for speed
        # Create mask of even/odd indices (starting from 0)
        odd = np.arange(n + 1) % 2
        weights = 1 + 3 * odd  # 1 for even, 4 for odd

        # Integral for each interval: (dx/3) * ( first + 4*mid + last )
        # Compute cumulative sum of weighted values
        # We need cumulative sum over the extended array weighted
        w_x = weights * extended
        cum = np.cumsum(w_x, axis=0)

        # Subtract the last weight (the last point is not part of any full interval)
        cum -= weights[-1] * extended[-1]

        # Scale by dx/3 and shift to match cumulative integration
        result = (dx / 3.0) * cum[:-1]  # remove the final extra zero

        # Permute back to original shape
        return np.swapaxes(result, -1, 0)

    def solve(self, problem: dict) -> NDArray:
        """
        Compute the cumulative integral along the last axis of the multi‑dimensional array using
        a hand‑crafted, vectorised implementation of Simpson's rule.
        """
        y2 = problem['y2']
        dx = problem['dx']
        return self._cumulative_simpson(y2, dx)