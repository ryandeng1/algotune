import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        # Use NumPy's built‑in outer product routine (fast C implementation)
        return np.outer(vec1, vec2)