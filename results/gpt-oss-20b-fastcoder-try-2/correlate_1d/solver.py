import numpy as np
class Solver:
    def __init__(self):
        self.mode = 'full'          # 'full' or 'valid'

    # pragma: no cover
    def solve(self, problem: list) -> list:
        """
        Compute the 1‑D correlation for each input pair.
        For mode 'valid', only pairs where `b` does not exceed `a` in length
        are processed.  The default mode is `'full'`.

        Parameters
        ----------
        problem : list[tuple[np.ndarray, np.ndarray]]
            List of tuples containing the two 1‑D arrays to be correlated.

        Returns
        -------
        list[np.ndarray]
            Correlation results for each valid pair.
        """
        mode = self.mode
        res = []

        if mode == 'valid':
            for a, b in problem:
                if b.shape[0] <= a.shape[0]:
                    res.append(np.correlate(a, b, mode='valid'))
        else:  # full
            for a, b in problem:
                res.append(np.correlate(a, b, mode='full'))

        return res