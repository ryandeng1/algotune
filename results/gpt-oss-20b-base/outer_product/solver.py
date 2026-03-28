from typing import Any
import numpy as np
from numba import njit

@njit
def _outer_product(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    # Simple element‑wise multiplication with broadcasting
    return vec1[:, None] * vec2

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        return _outer_product(vec1, vec2)