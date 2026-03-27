import numpy as np
from typing import List, Tuple

class Solver:
    """
    Compute the 1D correlation for each valid pair in the problem list.

    For mode 'valid', process only pairs where the length of the second array does not exceed the first.
    Return a list of 1D arrays representing the correlation results.
    """

    def __init__(self, mode: str = "valid") -> None:
        self.mode = mode

    def solve(self, problem: List[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
        results = []
        mode = self.mode
        for a, b in problem:
            if mode == "valid" and b.shape[0] > a.shape[0]:
                continue
            # np.correlate supports only 'valid', 'same', 'full'
            if mode not in {"valid", "same", "full"}:
                raise ValueError(f"Unsupported mode: {mode}")
            results.append(np.correlate(a, b, mode=mode))
        return results