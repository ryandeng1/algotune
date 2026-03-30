# solver.py
import numpy as np
from scipy import signal


class Solver:
    """Fast correlation solver."""

    def __init__(self):
        # Pre‑defined modes optimise branch lookup during execution
        self.mode = 'full'  # can be changed externally if needed
        self._use_valid = self.mode == 'valid'

    def solve(self, problem: list) -> list:
        """
        Compute the 1D correlation for each pair in ``problem``.

        Parameters
        ----------
        problem : list of tuples
            Each tuple contains two 1‑D numpy arrays (or array‑like).

        Returns
        -------
        list of ndarray
            Correlation results for each valid pair.
        """
        mode = self.mode
        use_valid = self._use_valid

        # Local references for speed
        correlate = signal.correlate
        res = []

        for a, b in problem:
            if use_valid and b.shape[0] > a.shape[0]:
                # Skip pairs that violate the 'valid' constraint
                continue
            res.append(correlate(a, b, mode=mode))

        return res