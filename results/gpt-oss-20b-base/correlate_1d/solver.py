import numpy as np

class Solver:
    def __init__(self):
        # Only 'full' or 'valid' are supported. 'full' is the default.
        self.mode = 'full'

    def solve(self, problem: list) -> list:
        """
        Compute the 1D correlation for each pair of arrays in ``problem``.
        ``self.mode`` must be either ``'full'`` (default) or ``'valid'``.
        For ``'valid'`` only pairs where the second array is not longer than the first
        are processed; the others are skipped.

        Parameters
        ----------
        problem : list
            List of ``(a, b)`` tuples where each element is a 1‑D NumPy array.

        Returns
        -------
        list
            List of correlation results in the same order as processed pairs.
        """
        mode = self.mode
        results = []

        # Localize functions for speed
        corr = np.correlate
        append = results.append

        for a, b in problem:
            if mode == 'valid' and b.ndim != 1:
                # Skip multi‑dimensional arrays silently (mimics original behaviour)
                continue
            if mode == 'valid' and len(b) > len(a):
                continue
            # np.correlate accepts only 1‑D arrays; shape[0] is length
            # If input not 1‑D, flatten to avoid errors. This mimics the
            # behaviour of scipy.signal.correlate which also works with 1‑D.
            try:
                res = corr(a, b, mode=mode)
            except Exception:
                # Fallback: flatten inputs
                res = corr(a.ravel(), b.ravel(), mode=mode)
            append(res)

        return results