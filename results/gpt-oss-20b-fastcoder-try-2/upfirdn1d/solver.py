import numpy as np
from scipy import signal
from typing import List, Tuple

class Solver:
    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray, int, int]]) -> List[np.ndarray]:
        """
        Compute the upfirdn operation for each problem definition in the list.
        """
        results = []
        for h, x, up, down in problem:
            # Use scipy's efficient implementation directly
            results.append(signal.upfirdn(h, x, up=up, down=down))
        return results