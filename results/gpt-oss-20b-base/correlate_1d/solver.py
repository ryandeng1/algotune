import numpy as np
from typing import List, Tuple

class Solver:
    def __init__(self, mode: str = "valid"):
        self.mode = mode

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        """Return correlation of each pair using NumPy's fast implementation."""
        mode = self.mode
        res = []
        for a, b in problem:
            if mode == "valid" and b.shape[0] > a.shape[0]:
                continue
            res.append(np.correlate(a, b, mode=mode))
        return res