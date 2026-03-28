from typing import Any
import numpy as np

class Solver:

    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        vec1, vec2 = problem
        outer_product = np.outer(vec1, vec2)
        return outer_product
