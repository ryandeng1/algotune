import numpy as np
from numpy import ndarray

class Solver:
    """Optimised 1‑D correlation solver."""

    def __init__(self, mode: str = "full"):
        """
        Parameters
        ----------
        mode : str
            Correlation mode: 'full', 'valid' or 'same'. Default is 'full'.
        """
        if mode not in {"full", "valid", "same"}:
            raise ValueError(f"Unsupported mode {mode!r}")
        self.mode = mode

    def solve(self, problem: list) -> list:
        """
        Compute the 1‑D correlation for each pair in *problem*.

        Parameters
        ----------
        problem : list
            List of tuples, each containing two 1‑D arrays (numpy.ndarray or list of floats).

        Returns
        -------
        list
            List of correlation results (numpy.ndarray) for each pair that satisfies
            the 'valid' length check (if mode is 'valid').
        """
        results = []
        # Local variable for speed
        mode = self.mode
        for a, b in problem:
            # Convert to numpy arrays if necessary
            if not isinstance(a, ndarray):
                a = np.asarray(a, dtype=np.float64)
            if not isinstance(b, ndarray):
                b = np.asarray(b, dtype=np.float64)

            # Skip invalid 'valid' pairs
            if mode == "valid" and b.shape[0] > a.shape[0]:
                continue

            # Use numpy.correlate which is fast and avoids extra imports
            res = np.correlate(a, b, mode=mode)
            results.append(res)
        return results
