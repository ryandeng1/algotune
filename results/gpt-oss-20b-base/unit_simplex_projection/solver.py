# solver.py
import numpy as np
from typing import Any, Dict, List


class Solver:
    """
    Computes the Euclidean projection of a vector onto the probability simplex.
    """

    def solve(self, problem: Dict[str, Any]) -> Dict[str, List[float]]:
        """
        Solve the Euclidean projection onto the probability simplex for the input vector `y`.

        Parameters
        ----------
        problem
            Dictionary containing the key 'y' with a 1‑D array‑like object.

        Returns
        -------
        dict
            Contains a single key 'solution' whose value is a Python list with the projected
            vector in the original order of `y`.
        """
        # Convert to a 1‑D NumPy array of floats (in‑place if possible)
        y = np.asarray(problem["y"], dtype=np.float64).ravel()

        # Sort in descending order once
        sorted_y = np.sort(y)[::-1]

        # Compute cumulative sums of the sorted vector and subtract 1 once
        cumsum_y = np.cumsum(sorted_y) - 1.0

        # Pre‑compute the divisor array (1, 2, …, n) to avoid repeated allocation
        n = sorted_y.size
        div = np.arange(1, n + 1, dtype=np.float64)

        # Find the largest index where sorted_y > cumsum_y / div
        # Using a vectorized comparison; the last True index gives rho
        comparison = sorted_y > cumsum_y / div
        rho = np.argmax(comparison[::-1])  # equivalent to finding the last True
        rho = n - rho - 1                    # adjust index after reversal

        # Compute the threshold theta
        theta = cumsum_y[rho] / (rho + 1.0)

        # Apply the projection formula efficiently using np.maximum
        x = np.maximum(y - theta, 0.0)

        # Return a plain Python list to avoid exposing NumPy arrays to the caller
        return {"solution": x.tolist()}