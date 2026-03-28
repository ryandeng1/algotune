import numpy as np
from typing import List, Tuple, Sequence

class Solver:
    def __init__(self):
        self.mode = 'full'

    def _validate_input(self, a: ArrayLike, b: ArrayLike) -> bool:
        """
        In 'valid' mode skip pairs where |b| > |a|.
        """
        return self.mode != 'valid' or a.shape[0] >= b.shape[0]

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Compute the 1‑D cross‑correlation for each pair in *problem*.

        Parameters
        ----------
        problem : list of (array, array) tuples
            Each tuple contains two 1‑D NumPy arrays.

        Returns
        -------
        list of np.ndarray
            Correlation results for every pair that satisfies the mode constraint.
        """
        results: List[np.ndarray] = []

        for a, b in problem:
            if not self._validate_input(a, b):
                continue

            # np.correlate is a thin wrapper around the highly optimised FFT or direct method
            results.append(np.correlate(a, b, mode=self.mode))

        return results