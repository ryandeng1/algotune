from typing import Any, List, Tuple
import numpy as np

class Solver:
    def __init__(self) -> None:
        # 'valid' or 'full'
        self.mode = 'full'

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """
        Compute the 1D correlation for each valid pair in the problem list.

        For mode 'valid', process only pairs where the length of the second array does not exceed the first.
        Return a list of 1D arrays representing the correlation results.
        """
        results = []
        mode = self.mode

        for a, b in problem:
            if mode == 'valid' and b.size > a.size:
                continue

            # numpy.correlate is typically faster than scipy.signal.correlate for 1‑D data
            # and is sufficient for this use case.
            results.append(np.correlate(a, b, mode=mode))
        return results