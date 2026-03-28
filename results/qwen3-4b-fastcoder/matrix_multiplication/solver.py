from typing import Any
import numpy as np

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[list[float]]:
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        C = A @ B
        return C.tolist()