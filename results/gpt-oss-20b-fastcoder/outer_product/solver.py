from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        # Broadcasting multiplication is typically faster than np.outer
        return vec1[:, None] * vec2