import numpy as np
from typing import Tuple


class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return the outer product of two 1‑D numpy arrays."""
        vec1, vec2 = problem
        # np.multiply.outer is a thin wrapper that avoids an intermediate copy
        return np.multiply.outer(vec1, vec2)