from typing import Any
import numpy as np


class Solver:
    def solve(self, problem: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        return np.outer(*problem)