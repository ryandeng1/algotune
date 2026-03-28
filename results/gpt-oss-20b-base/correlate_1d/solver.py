from typing import List, Tuple, Any
import numpy as np

class Solver:
    def __init__(self):
        self.mode = 'full'  # could be 'valid', 'same', or 'full'

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Compute the 1D correlation for each valid pair in the problem list.

        Parameters
        ----------
        problem : List[Tuple[np.ndarray, np.ndarray]]
            A list of tuples, each containing two 1D numpy arrays.

        Returns
        -------
        List[np.ndarray]
            A list of 1D arrays representing the correlation results.
        """
        mode = self.mode
        res = []
        # Local variable for performance
        _np_correlate = np.correlate
        for a, b in problem:
            # Skip if mode is 'valid' and b is longer than a
            if mode == 'valid' and b.shape[0] > a.shape[0]:
                continue
            res.append(_np_correlate(a, b, mode))
        return res